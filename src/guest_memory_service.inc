// SPDX-License-Identifier: GPL-3.0-or-later
// ---- The guest's memory as a service: shared until written (included by memory_pages.cpp) ----
//
// After Gregory Price's "A Compressed RAM Service" (Linux Plumbers 2026, mm/cram.c): memory that
// is read where it lies and promoted when it is written. There the device compresses; here the
// emulator is the device, and the two forms a processor can read without a fault are a slot that
// shares the block of zeroes and a slot that shares a block with another slot of the same bytes.
//
//   Zero    the slot shows the shared block of zeroes, read-only
//   Own     the slot has a block of its own, read-write
//   Shared  the slot shows a block it shares with other slots of the same bytes, read-only
//   Fine    the slot is 128 pages of 16 KiB, read-only: its pages of zeroes show the zero block, the
//           others a compact block that holds only them (a written slot with many pages of zeroes);
//           a write makes it whole again, and it waits longer each time before it may be Fine again
//   Packed  the slot's bytes are LZ4- or zstd-packed in the service's store, its block is gone; any
//           access faults and the data is unpacked into a block of its own
//   Probe   an Own slot that keeps its block but can be neither read nor written (armed): the first
//           touch of either kind faults and makes it Own again, so one that stays Probe has not been
//           touched. EDEN_GUEST_SWAP=1 measures only this
//   Out     a Probe slot untouched for long: its bytes are in a file on the internal SSD, its block
//           is gone, and its address space is reserved; any touch faults and reads it back
//
// A write to a Zero or Shared slot faults; HandleGuestWriteFault gives the slot a block of its
// own (zeroed, or a copy) and the write repeats. So memory is no longer taken when the guest's
// kernel hands pages out, only when they are written; a slot the guest's kernel clears or frees
// whole goes back to the zero block; and a scanner gives back slots that hold only zeroes and
// lets slots of the same bytes share one block. A slot is locked (bit 7 of its state) while it
// changes; a writer that faults on a locked slot waits for it.
//
// 64-bit games only: a 32-bit game's fastmem window aliases the backing, which takes it whole.

extern "C" std::size_t eden_memory_recover_try(std::size_t bytes) noexcept;
#ifdef PS5_NATIVE
// The Eden heap's empty arenas and spaces (headless/heap_arenas.inc), asked from a write fault.
extern "C" [[gnu::weak]] std::size_t eden_heap_trim(void);
#endif

namespace {
enum : std::uint8_t {
    SlotZero = 0, SlotOwn = 1, SlotShared = 2, SlotFine = 3, SlotPacked = 4, SlotProbe = 5, SlotOut = 6,
    SlotLock = 0x80, SlotMask = 0x7f
};
constexpr std::size_t FinePage = 16384, FinePages = SparseSlot / FinePage;
// A written slot becomes Fine with at least this many pages of zeroes (512 KiB given back).
constexpr std::size_t FineMinZeroPages = 32;
// A slot made whole again by a write waits 1, 2, 4 ... 1024 scan passes before it may be Fine again.
constexpr unsigned FineBackoffMax = 10;

// Duplicate sharing is off unless asked for (dev-settings cram_dedup=on): on the console it found a
// dozen 2 MiB pairs that were written apart within a minute, and with it on a large game held 20 FPS
// instead of 30 (the GPU worker waited for every fence, once a frame) where it held 30 without.
std::atomic<bool> service_requested{true}, scan_requested{true}, dedup_requested{false}, fine_requested{true};
std::atomic<SparseRange*> service_range{nullptr}; // the guest range the service runs for
std::atomic<SparseRange*> probe_range{nullptr};   // the self-test's range
std::atomic<int> service_state{0};                // 0 not tested, 1 works here, 2 does not
const char* service_refused = "";

struct Counters {
    std::atomic<std::uint64_t> faults{0}, promotions{0}, copies{0}, waits{0};
    std::atomic<std::uint64_t> returns_clear{0}, returns_free{0}, returns_scan{0}, dedup{0};
    std::atomic<std::uint64_t> reserve_used{0}, host_reserve_used{0}, failures{0}, passes{0}, scan_ns{0};
    std::atomic<std::uint64_t> fined{0}, unfined{0}, fine_failures{0};
};
Counters counters;

// ---- The 64-bit fastmem window's aliases (src/host_memory.cpp, Window64; backported from ProsperoEden) ----
// A window region (2 MiB) shows a slot's block from a JIT fault until the slot's block changes. Every
// change of a slot's block or protection (a promotion, a return, a scan's fence) first drops its
// aliases, under the slot's lock: a stale alias would reach memory the slot gave up. An entry is the
// window address, plus AliasReadOnly when it shows the block read-only, or AliasSmall when it shows
// 16 KiB pieces of it rather than the whole block.
constexpr unsigned AliasesPerSlot = 4;
constexpr std::uintptr_t AliasReadOnly = 1, AliasSmall = 2, AliasFlags = AliasReadOnly | AliasSmall;
std::atomic<std::uint64_t> aliases_made{0}, aliases_dropped{0}, aliases_refused{0};
// The window region at `at` shows nothing again: a reservation straight over the mapping.
void VoidWindowRegion([[maybe_unused]] std::uintptr_t at) noexcept {
#ifdef PS5_NATIVE
    void* address = reinterpret_cast<void*>(at);
    if (sceKernelReserveVirtualRange(&address, SparseSlot, MAP_FIXED, SparseSlot) == 0 && address == reinterpret_cast<void*>(at))
        return;
    address = reinterpret_cast<void*>(at);
    if (munmap(address, SparseSlot) != 0 || sceKernelReserveVirtualRange(&address, SparseSlot, MAP_FIXED, SparseSlot) != 0 ||
        address != reinterpret_cast<void*>(at))
        std::abort(); // a window region left mapped would expose memory the slot gives up
#else
    if (mmap(reinterpret_cast<void*>(at), SparseSlot, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0) ==
        MAP_FAILED)
        std::abort();
#endif
}
// Slot locked.
void DropSlotAliases(SparseRange& range, std::size_t slot) noexcept {
    if (!range.guest || !range.aliases) return;
    for (unsigned i = 0; i < AliasesPerSlot; ++i)
        if (auto& alias = range.aliases[slot * AliasesPerSlot + i]) {
            VoidWindowRegion(alias & ~AliasFlags);
            alias = 0;
            aliases_dropped.fetch_add(1, std::memory_order_relaxed);
        }
}
void ResetCounters() noexcept {
    for (auto* counter : {&counters.faults, &counters.promotions, &counters.copies, &counters.waits,
                          &counters.returns_clear, &counters.returns_free, &counters.returns_scan, &counters.dedup,
                          &counters.reserve_used, &counters.host_reserve_used, &counters.failures, &counters.passes, &counters.scan_ns,
                          &counters.fined, &counters.unfined, &counters.fine_failures})
        counter->store(0, std::memory_order_relaxed);
}

// Freed pages cleared (GuestMemoryFreedPart, below).
std::atomic<std::uint64_t> freed_part_calls{0}, freed_part_bytes{0};
// Idle blocks RecoverMemory gave back (GiveBackIdleBlocks).
std::atomic<std::uint64_t> idle_given_back{0}, idle_give_back_calls{0};

bool LockSlot(SparseRange& range, std::size_t slot, bool count_wait = false) noexcept {
    auto& state = range.state[slot];
    bool waited = false;
    for (;;) {
        std::uint8_t value = state.load(std::memory_order_relaxed);
        if (!(value & SlotLock) &&
            state.compare_exchange_weak(value, value | SlotLock, std::memory_order_acquire, std::memory_order_relaxed))
            break;
        waited = true;
        sched_yield();
    }
    if (waited && count_wait) counters.waits.fetch_add(1, std::memory_order_relaxed);
    return waited;
}
// Locks a slot only if no one holds it: the graphics caches' discard runs on the thread that
// draws, and a write fault holds a slot's lock while it waits for a block (WaitForBlock, up to
// 2 s), so it must never be waited for there. False means the slot was left alone this pass.
bool TryLockSlot(SparseRange& range, std::size_t slot) noexcept {
    auto& state = range.state[slot];
    std::uint8_t value = state.load(std::memory_order_relaxed);
    if (value & SlotLock) return false;
    return state.compare_exchange_strong(value, value | SlotLock, std::memory_order_acquire,
                                         std::memory_order_relaxed);
}
std::size_t FineKept(SparseRange& range, std::size_t slot) noexcept {
    return range.fine[slot] & 0xff;
}
std::uint8_t SlotState(SparseRange& range, std::size_t slot) noexcept {
    return range.state[slot].load(std::memory_order_acquire) & SlotMask;
}
// Sets the state and lets the slot go.
void UnlockSlot(SparseRange& range, std::size_t slot, std::uint8_t to) noexcept {
    // Packed and Out: no block; Probe keeps the Own slot's block, with no access.
    range.owned[slot].store(to != SlotZero && to != SlotPacked && to != SlotOut, std::memory_order_release);
    range.state[slot].store(to, std::memory_order_release);
}
std::uintptr_t SlotAddress(SparseRange& range, std::size_t slot) noexcept {
    return range.begin.load(std::memory_order_relaxed) + slot * SparseSlot;
}

// ---- Cold slots and the swap file (EDEN_GUEST_SWAP, off unless asked for) ----
//
// The processor's accessed bits are out of reach here, so a slot shows it is cold by what does not
// happen: it is armed (Probe) a few at a time, and the first touch, read or write, comes back
// through the fault. A slot that stays Probe for EDEN_GUEST_SWAP_COLD_S goes out: to the swap file
// (`Out`), its block given back to the pool; a touch reads it back into a fresh block. This is the
// only route that moves the live, poorly-compressing guest heap out of the 12,224 MiB pool: LZ4
// packs Zelda Ocarina of Time's memory at about 0.83 (docs/MKW-MEMORY-LEAK.md measured 0.61 for
// Mario Kart World), so packing could not take it.
//
//   EDEN_GUEST_SWAP=1        arm and count cold slots; nothing leaves memory (measure)
//   EDEN_GUEST_SWAP=2        also write cold slots to the file while the pool is low
//   EDEN_GUEST_SWAP_FILE     the file (default /data/eglantineps5/guest-swap.bin, internal SSD)
//   EDEN_GUEST_SWAP_MIB      the most that may be in the file (default 2048)
//   EDEN_GUEST_SWAP_COLD_S   how long a slot stays untouched to count as cold (default 30)
//   EDEN_GUEST_SWAP_LOW_MIB  the pool's largest free block under which cold slots go out (default 1024)
//   EDEN_GUEST_SWAP_AHEAD_MIB=<n>  also send cold slots out while the pool's largest free block is
//                            under n MiB, before it is low: Ocarina's pool fell 2.4 GiB -> 0.2 GiB
//                            within 20 s at its first 3D frame, faster than cold slots leave.
//   EDEN_GUEST_SWAP_ALL_POOLS=1  arm every Own slot, not only the game's pool (the default, 0,
//                            arms only the game's pool, whose place the core reports)
// The real GPU never sees this memory: no mapping of it has GPU flags and no Vulkan import takes it.
// The emulated GPU, the JIT and the emulator read it with the processor, and those faults come here.
// A system call given a pointer into a Probe or Out slot is the one path that does not fault here;
// the core copies IPC read buffers while this runs (hle_ipc.cpp, eden_guest_swap_active) and says
// where the game's pool lies (kernel.cpp, eden_guest_swap_pool); until it has, nothing is armed.
constexpr unsigned SwapTickMs = 2000;
constexpr std::uint32_t SwapRearmTicks = 8;    // a touched slot stays Own this long (16 s)
constexpr std::size_t SwapArmShare = 8;        // an eighth of the slots is looked at each tick
constexpr std::size_t SwapOutPerTick = 64;     // at most 64 slots (128 MiB) a tick
constexpr std::uint64_t SwapBackInPerTick = 8; // more reads back than this in a tick: a thrash guard
constexpr std::uint32_t SwapMostPlaces = 4096; // 8 GiB of file
constexpr std::size_t SwapAges = 8;
std::atomic<int> swap_mode{0}; // 0 off, 1 measure (arm/touch), 2 file
std::atomic<std::uint32_t> swap_tick{SwapRearmTicks + 1};
std::uint32_t swap_cold_ticks = 15;       // EDEN_GUEST_SWAP_COLD_S, 30 s by default
std::uint32_t swap_low_bytes_mib = 1024;  // EDEN_GUEST_SWAP_LOW_MIB
std::size_t swap_ahead_bytes = 0;         // EDEN_GUEST_SWAP_AHEAD_MIB, 0 = only while low
bool swap_all_pools = false;              // EDEN_GUEST_SWAP_ALL_POOLS=1: every Own slot, not only the game's pool
std::atomic<std::uint64_t> swap_pool_first{0}, swap_pool_end{0}; // the game's pool, in slots
std::atomic<std::size_t> swap_pool_largest{0}; // the pool's largest free block at the last tick
std::uint32_t swap_places = 1024;              // places in the file (EDEN_GUEST_SWAP_MIB / 2 MiB)
const char* swap_path = "/data/eglantineps5/guest-swap.bin";
const char* swap_refused = "";
bool swap_checked = false; // the self-test ran this process
[[maybe_unused]] int swap_fd = -1;
std::uint32_t swap_free[SwapMostPlaces];
std::uint32_t swap_free_count = 0, swap_next_place = 0;
std::atomic_flag swap_lock = ATOMIC_FLAG_INIT;
struct SwapCounters {
    std::atomic<std::uint64_t> armed{0}, touched{0}, out{0}, in{0}, out_ns{0}, in_ns{0}, in_most_ns{0}, failures{0};
    std::atomic<std::uint64_t> age[SwapAges]{}; // touches by how long the slot was armed: under 2^n ticks
    std::atomic<std::uint64_t> returned_armed{0}; // slots given back while they were Probe
};
SwapCounters swap_counters;
// Slots that ever went out (the core's EDEN_NX2_JUNK_TRACE asks through eden_guest_swap_address_state):
// one bit a slot, 16384 slots cover a 32 GiB range.
constexpr std::size_t SwapEverOutSlots = 16384;
std::atomic<std::uint64_t> swap_ever_out[SwapEverOutSlots / 64]{};
std::atomic<std::uint64_t> swap_given_back{0};   // all the memory the swap has given back
[[maybe_unused]] std::atomic<std::uint64_t> swapped_to_file{0}; // slots written to the file
// Allocations failing now (eden_guest_memory_urgent): until then the swapper ticks at once and sends
// to the file whatever must go, and reads coming back do not stop it.
std::atomic<std::uint64_t> swap_urgent_until_ns{0};
std::atomic<bool> swap_urgent_wake{false};
// Reads of the file waiting now: the writer gives way to them (see SwapWrite).
[[maybe_unused]] std::atomic<int> swap_readers{0};

[[maybe_unused]] std::uint64_t NowNs() noexcept {
    timespec now{};
    clock_gettime(CLOCK_MONOTONIC, &now);
    return static_cast<std::uint64_t>(now.tv_sec) * 1000000000ull + static_cast<std::uint64_t>(now.tv_nsec);
}
void SwapReport(SparseRange& range) noexcept; // below, with the swapper
void Spin(std::atomic_flag& flag) noexcept;   // below, with the blocks
// A place in the file, the lowest never used first: a file on exFAT only ever grows at its end.
[[maybe_unused]] bool TakeSwapPlace(std::uint32_t* place) noexcept {
    Spin(swap_lock);
    bool ok = true;
    if (swap_free_count != 0) *place = swap_free[--swap_free_count];
    else if (swap_next_place < swap_places) *place = swap_next_place++;
    else ok = false;
    swap_lock.clear(std::memory_order_release);
    return ok;
}
void GiveSwapPlace(std::uint32_t place) noexcept {
    Spin(swap_lock);
    if (swap_free_count < SwapMostPlaces) swap_free[swap_free_count++] = place;
    swap_lock.clear(std::memory_order_release);
}
#ifdef PS5_NATIVE
// A read of the file has the writer give way (SwapWrite) and is timed: the caller is a fault handler,
// and Ocarina's reads back took 70-80 ms on average and up to 2.9 s queued behind the writer.
bool SwapRead(void* to, std::uint32_t place) noexcept {
    swap_readers.fetch_add(1, std::memory_order_acq_rel);
    auto* at = static_cast<std::uint8_t*>(to);
    std::size_t done = 0;
    bool ok = true;
    while (done < SparseSlot) {
        const auto got = pread(swap_fd, at + done, SparseSlot - done, static_cast<off_t>(place) * SparseSlot + done);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) {
            ok = false;
            break;
        }
        done += static_cast<std::size_t>(got);
    }
    swap_readers.fetch_sub(1, std::memory_order_acq_rel);
    return ok;
}
// The writer waits while reads are waiting (at most 1 s at a time, so writes never stop for good),
// and writes in pieces of WritePiece, so a read waits for one piece at most.
constexpr std::size_t WritePiece = std::size_t{256} << 10;
void GiveWayToReads() noexcept {
    for (int i = 0; i < 1000 && swap_readers.load(std::memory_order_acquire) != 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
}
bool SwapWrite(const void* from, std::uint32_t place) noexcept {
    const auto* at = static_cast<const std::uint8_t*>(from);
    std::size_t done = 0;
    while (done < SparseSlot) {
        GiveWayToReads();
        const std::size_t piece = std::min(WritePiece, SparseSlot - done);
        const auto put = pwrite(swap_fd, at + done, piece, static_cast<off_t>(place) * SparseSlot + done);
        if (put < 0 && errno == EINTR) continue;
        if (put <= 0) return false;
        done += static_cast<std::size_t>(put);
    }
    return true;
}
#endif

// ---- Blocks: the pool, a reserve for when the pool says no, and shared blocks' references ----
//
// The reserve is the guest's headroom in the pool, held as blocks of its own: 1 GiB that a game's
// writes take when the pool has no block left. It used to be 32 MiB here, plus 1 GiB that the
// graphics caches only counted as taken (headless/performance.cpp): that 1 GiB stayed in the pool,
// where Eden's heap, the driver and the caches themselves could take it, and where 2 MiB blocks
// could no longer be cut from what was left. Tears of the Kingdom's writes found no block four
// times with the pool's largest free block at 1 or 2 MiB. Held here, nothing else can take it or
// cut it up; the caches see the pool as it is, less what is not held yet, and evict as soon as a
// burst of writes has used half of it (performance.h ViewOfPool). It is topped up every 100 ms
// (ReserveKeeper) and given back with the guest's memory when the session ends.
constexpr std::size_t ReserveBlocks = 256;      // 512 MiB (diagnostic62 shared-pool test)
constexpr std::size_t ReserveAtStart = 16;      // taken as the service starts; the keeper takes the rest
std::int64_t reserve[ReserveBlocks];
bool PackEnabled() noexcept; // packing (below)
// The reserve kept, in blocks: EDEN_CRAM_RESERVE_MIB (up to 512), else 128 MiB while packing runs
// (packing gives blocks back when the pool is short; MKW used none of the 512 MiB), else 512 MiB.
std::size_t ReserveTarget() noexcept {
    static const std::size_t blocks = [] {
        const char* value = std::getenv("EDEN_CRAM_RESERVE_MIB");
        const std::size_t mib = value != nullptr ? std::strtoull(value, nullptr, 10) : PackEnabled() ? 128 : 512;
        return std::max<std::size_t>(1, std::min(ReserveBlocks, mib / (SparseSlot >> 20)));
    }();
    return blocks;
}
std::atomic<std::size_t> reserve_count{0};
std::atomic_flag reserve_lock = ATOMIC_FLAG_INIT;
constexpr std::size_t SharedBlocks = 4096;
std::int64_t shared_block[SharedBlocks];
int shared_refs[SharedBlocks];
std::atomic_flag shared_lock = ATOMIC_FLAG_INIT;

void Spin(std::atomic_flag& flag) noexcept {
    while (flag.test_and_set(std::memory_order_acquire)) sched_yield();
}

// ---- Packed slots: an Own slot unwritten for a while, compressed while the pool is short ----
// LZ4 first (fast to unpack); a slot LZ4 packs little is tried with zstd level 1 (EDEN_SWAP_ZSTD,
// on the console when zstd.h is there), which packs Ocarina's stubbornly-compressing memory tight
// enough to keep it in memory instead of the swap file.
// The slot maps the zero block with no access and its block goes back; the next access of any
// kind faults, and the fault handler gives it a block and unpacks it there (UnpackSlot). Packed
// data lives in the heap. A buffer the fault handler is done with is freed by the scanner (no
// free in a signal handler): it goes on a list through its first word.
struct PackedSlot {
    void* data = nullptr;
    std::uint32_t size = 0;
};
std::atomic<PackedSlot*> packed_slots{nullptr}; // per slot of the guest's range, made by the scanner
// Diagnostic (EDEN_CRAM_PACK_KEEP=1): a packed slot keeps its block, and the unpack compares what it
// unpacked with it. A difference means the slot's data changed while it was packed (written through a
// mapping other than the slot's own). Saves no memory: for finding what packing breaks.
bool PackKeep() noexcept {
    static const bool on = [] {
        const char* value = std::getenv("EDEN_CRAM_PACK_KEEP");
        return value != nullptr && value[0] == '1';
    }();
    return on;
}
std::int64_t* pack_kept = nullptr; // per slot: the block a packed slot kept, -1 none
// Pages of each slot the GPU's caches hold (eden_guest_gpu_cached, from the core's device memory
// manager): such a slot is not packed. With packing, Mario Kart World's emulated GPU read impossible
// draw arguments (1.8-2.9 billion indices) and the GPU hung, only in sessions where the GPU thread
// itself touched packed slots; the slots' data came back intact, so what the GPU reads through its
// caches and their ordering must not meet a packed slot. That left MKW ~86 MiB to pack (it maps
// nearly all its memory for the GPU), and it still ran out; the impossible draws come without
// packing too. So such slots are packed by default; EDEN_CRAM_PACK_GPU=0 keeps them out.
std::atomic<std::atomic<std::int32_t>*> gpu_cached_table{nullptr};
std::size_t gpu_cached_slots = 0;
std::atomic<std::uint64_t> pack_gpu_skips{0}, pack_gpu_marked{0};
// Slots a GPU thread (the command reader "GPU", or "GPUBack") ever had to unpack: never packed again.
// The caches' count misses what the command reader reads (command lists, draw arguments), and one
// such unpack was enough for impossible draws.
std::atomic<std::uint8_t>* gpu_touched = nullptr;
// Diagnostic: which threads touched packed slots (unpacks by thread name, 16 names at most).
struct Unpacker {
    char name[24];
    std::atomic<std::uint32_t> count;
};
Unpacker unpackers[16];
std::atomic_flag unpackers_lock = ATOMIC_FLAG_INIT;
void CountUnpacker(std::size_t slot) noexcept;
std::atomic<std::uint64_t> pack_keep_compared{0}, pack_keep_differed{0};
std::atomic<std::uint64_t> packed_count{0}, packed_bytes{0}, packs{0}, unpacks{0}, pack_skips{0};
// A packed slot's size: the top bit says zstd packed it (LZ4 otherwise). Both pack below 2 MiB
// (PackMaxRatio), so the bit is free.
constexpr std::uint32_t PackedZstdBit = 0x80000000u;
std::uint32_t PackedStoredBytes(std::uint32_t stored) noexcept { return stored & ~PackedZstdBit; }
#ifdef EDEN_SWAP_ZSTD
// One packing context, used under pack_mutex as the LZ4 buffer is; unpacking runs in fault
// handlers on any thread, so it takes one of a few contexts made in static memory at the start
// (no allocation in a fault). EDEN_GUEST_SWAP_ZSTD=0 turns it off.
ZSTD_CCtx* zstd_pack = nullptr;
constexpr unsigned ZstdUnpackers = 8;
constexpr std::size_t ZstdUnpackerBytes = std::size_t{256} << 10;
alignas(64) unsigned char zstd_unpacker_memory[ZstdUnpackers][ZstdUnpackerBytes];
ZSTD_DCtx* zstd_unpackers[ZstdUnpackers]{};
std::atomic_flag zstd_unpacker_busy[ZstdUnpackers] = {};
std::atomic<bool> zstd_ready{false};
std::atomic<std::uint64_t> packed_zstd{0};

// Once, where allocating is fine (the service's start).
void ZstdStart() noexcept {
    const char* off = std::getenv("EDEN_GUEST_SWAP_ZSTD");
    if ((off != nullptr && off[0] == '0') || zstd_ready.load(std::memory_order_relaxed)) return;
    if (ZSTD_estimateDCtxSize() > ZstdUnpackerBytes) return;
    for (unsigned i = 0; i < ZstdUnpackers; ++i) {
        zstd_unpackers[i] = ZSTD_initStaticDCtx(zstd_unpacker_memory[i], ZstdUnpackerBytes);
        if (zstd_unpackers[i] == nullptr) return;
    }
    zstd_pack = ZSTD_createCCtx();
    if (zstd_pack == nullptr) return;
    zstd_ready.store(true, std::memory_order_release);
}

// zstd-packed bytes back into 2 MiB; false if they do not come back whole. Waits for a context:
// the eight are only taken for the milliseconds an unpack takes.
bool ZstdUnpack(const void* from, std::size_t bytes, void* to) noexcept {
    if (!zstd_ready.load(std::memory_order_acquire)) return false;
    for (;;) {
        for (unsigned i = 0; i < ZstdUnpackers; ++i) {
            if (zstd_unpacker_busy[i].test_and_set(std::memory_order_acquire)) continue;
            const std::size_t got = ZSTD_decompressDCtx(zstd_unpackers[i], to, SparseSlot, from, bytes);
            zstd_unpacker_busy[i].clear(std::memory_order_release);
            return !ZSTD_isError(got) && got == SparseSlot;
        }
        sched_yield();
    }
}
#endif

// The packed store: packed data lives in 2 MiB blocks of direct memory of its own, filled one after
// another, outside Eden's heap. Packing then takes no heap memory, so it can run where the heap
// itself found no memory (eden_memory_recover_try), and leaves no slack in the heap. A block whose
// data all went back is given back by the scanner (StoreReclaim). When the pool has no block for
// it, the slot being packed gives its own. Taken and filled under pack_mutex; StoreRelease (the
// fault handler) only counts.
constexpr std::size_t StoreBlock = SparseSlot, StoreBlocks = 2048; // up to 4 GiB of packed data
std::uintptr_t store_base = 0;
std::int64_t store_physical[StoreBlocks];
std::uint32_t store_used[StoreBlocks];
std::atomic<std::uint32_t> store_live[StoreBlocks];
std::size_t store_current = StoreBlocks;
std::atomic<std::size_t> store_held{0};

#ifdef PS5_NATIVE
bool StoreStart() noexcept {
    if (store_base != 0) return true;
    void* address = reinterpret_cast<void*>(cpu_mapping_hint);
    if (sceKernelReserveVirtualRange(&address, StoreBlock * StoreBlocks, 0, StoreBlock) != 0) return false;
    store_base = reinterpret_cast<std::uintptr_t>(address);
    for (std::size_t i = 0; i < StoreBlocks; ++i) store_physical[i] = -1;
    return true;
}
// Room for `size` bytes (pack_mutex held). *donate: a block the caller gives up if the pool has
// none (-1 afterwards when it was taken).
void* StoreAlloc(std::size_t size, std::int64_t* donate) noexcept {
    if (store_current < StoreBlocks && store_used[store_current] + size <= StoreBlock) {
        void* at = reinterpret_cast<void*>(store_base + store_current * StoreBlock + store_used[store_current]);
        store_used[store_current] += static_cast<std::uint32_t>(size);
        store_live[store_current].fetch_add(static_cast<std::uint32_t>(size), std::memory_order_relaxed);
        return at;
    }
    std::size_t index = 0;
    while (index < StoreBlocks && store_physical[index] >= 0) ++index;
    if (index == StoreBlocks) return nullptr;
    std::int64_t physical = -1;
    if (eden_direct_low(StoreBlock, StoreBlock, 12, &physical) != 0) {
        if (donate == nullptr || *donate < 0) return nullptr;
        physical = std::exchange(*donate, -1);
    }
    if (!MapSlot(store_base + index * StoreBlock, PROT_READ | PROT_WRITE, physical)) {
        if (donate != nullptr && *donate < 0) *donate = physical;
        else (void)sceKernelReleaseDirectMemory(physical, StoreBlock);
        return nullptr;
    }
    store_physical[index] = physical;
    store_used[index] = static_cast<std::uint32_t>(size);
    store_live[index].store(static_cast<std::uint32_t>(size), std::memory_order_relaxed);
    store_current = index;
    store_held.fetch_add(StoreBlock, std::memory_order_relaxed);
    return reinterpret_cast<void*>(store_base + index * StoreBlock);
}
// Blocks whose data all went back (pack_mutex held).
void StoreReclaim() noexcept {
    // Nothing before the store has started: its tables are not set up (zero is not "no block"),
    // and mapping at store_base 0 would land on the app's own code (the self-test's range goes
    // back through here at start).
    if (store_base == 0) return;
    for (std::size_t i = 0; i < StoreBlocks; ++i) {
        if (store_physical[i] < 0 || i == store_current || store_live[i].load(std::memory_order_acquire) != 0) continue;
        if (!MapSlot(store_base + i * StoreBlock, PROT_NONE, zero_block)) continue;
        if (sceKernelReleaseDirectMemory(store_physical[i], StoreBlock) != 0) std::abort();
        store_physical[i] = -1;
        store_held.fetch_sub(StoreBlock, std::memory_order_relaxed);
    }
}
#endif
void StoreRelease(void* data, std::uint32_t size) noexcept {
    const std::size_t index = (reinterpret_cast<std::uintptr_t>(data) - store_base) / StoreBlock;
    store_live[index].fetch_sub(size, std::memory_order_release);
}
// Forgets a Packed slot's data (the slot is locked and leaves the Packed state).
void ForgetPacked(std::size_t slot) noexcept {
    PackedSlot* const table = packed_slots.load(std::memory_order_acquire);
    if (table == nullptr || table[slot].data == nullptr) return;
    packed_count.fetch_sub(1, std::memory_order_relaxed);
    packed_bytes.fetch_sub(PackedStoredBytes(table[slot].size), std::memory_order_relaxed);
    StoreRelease(table[slot].data, PackedStoredBytes(table[slot].size));
    table[slot] = {};
}

// Blocks packing freed, kept for unpacking (KeepForUnpack below).
constexpr std::size_t UnpackKeepBlocks = 256;
std::int64_t unpack_keep[UnpackKeepBlocks];
std::atomic<std::size_t> unpack_keep_count{0};
std::atomic_flag unpack_keep_lock = ATOMIC_FLAG_INIT;
#ifdef PS5_NATIVE
// A zeroed block from the reserve (the pool refused one).
bool ReserveBlock(std::int64_t* physical) noexcept {
    Spin(reserve_lock);
    const std::size_t count = reserve_count.load(std::memory_order_relaxed);
    const bool ok = count > 0;
    if (ok) {
        *physical = reserve[count - 1];
        reserve_count.store(count - 1, std::memory_order_relaxed);
    }
    reserve_lock.clear(std::memory_order_release);
    if (ok) counters.reserve_used.fetch_add(1, std::memory_order_relaxed);
    return ok;
}
// Zeroed blocks into the reserve until it holds `blocks` (at most ReserveBlocks) or the pool says no.
void FillReserve(std::size_t blocks = ReserveTarget()) noexcept {
    blocks = std::min(blocks, ReserveTarget());
    for (;;) {
        Spin(reserve_lock);
        const bool full = reserve_count.load(std::memory_order_relaxed) >= blocks;
        reserve_lock.clear(std::memory_order_release);
        if (full) return;
        std::int64_t physical = -1;
        if (!ZeroedBlock(&physical)) return;
        Spin(reserve_lock);
        const std::size_t count = reserve_count.load(std::memory_order_relaxed);
        if (count < blocks) {
            reserve[count] = physical;
            reserve_count.store(count + 1, std::memory_order_relaxed);
            physical = -1;
        }
        reserve_lock.clear(std::memory_order_release);
        if (physical >= 0 && sceKernelReleaseDirectMemory(physical, SparseSlot) != 0) std::abort();
    }
}
// Blocks packing freed, kept for unpacking only (an unpack overwrites the whole block, so they are
// not zeroed). A burst of packing under a short pool is followed by reads of some of those slots: a
// read whose unpack found no block waited in its fault handler for up to 2 s, and with dozens of them
// the threads the GPU's work waits on stalled until the driver declared the device lost (Mario Kart
// World: every GPU hang came seconds after ~1.3 GB was packed). EDEN_CRAM_UNPACK_KEEP_MIB (default
// 256, at most 512) sets how many are kept.
std::size_t UnpackKeepTarget() noexcept {
    static const std::size_t blocks = [] {
        const char* value = std::getenv("EDEN_CRAM_UNPACK_KEEP_MIB");
        const std::size_t mib = value != nullptr ? std::strtoull(value, nullptr, 10) : 256;
        return std::min(UnpackKeepBlocks, mib / (SparseSlot >> 20));
    }();
    return blocks;
}
bool KeepForUnpack(std::int64_t physical) noexcept {
    Spin(unpack_keep_lock);
    const std::size_t count = unpack_keep_count.load(std::memory_order_relaxed);
    const bool kept = count < UnpackKeepTarget();
    if (kept) {
        unpack_keep[count] = physical;
        unpack_keep_count.store(count + 1, std::memory_order_relaxed);
    }
    unpack_keep_lock.clear(std::memory_order_release);
    return kept;
}
bool UnpackBlock(std::int64_t* physical) noexcept {
    Spin(unpack_keep_lock);
    const std::size_t count = unpack_keep_count.load(std::memory_order_relaxed);
    const bool ok = count > 0;
    if (ok) {
        *physical = unpack_keep[count - 1];
        unpack_keep_count.store(count - 1, std::memory_order_relaxed);
    }
    unpack_keep_lock.clear(std::memory_order_release);
    return ok;
}
void EmptyUnpackKeep() noexcept {
    Spin(unpack_keep_lock);
    for (std::size_t i = 0; i < unpack_keep_count.load(std::memory_order_relaxed); ++i)
        if (sceKernelReleaseDirectMemory(unpack_keep[i], SparseSlot) != 0) std::abort();
    unpack_keep_count.store(0, std::memory_order_relaxed);
    unpack_keep_lock.clear(std::memory_order_release);
}
void EmptyReserve() noexcept {
    EmptyUnpackKeep();
    Spin(reserve_lock);
    for (std::size_t i = 0; i < reserve_count.load(std::memory_order_relaxed); ++i)
        if (sceKernelReleaseDirectMemory(reserve[i], SparseSlot) != 0) std::abort();
    reserve_count.store(0, std::memory_order_relaxed);
    reserve_lock.clear(std::memory_order_release);
}
#endif

// One more slot shows `physical`; the first one makes it a shared block. False when the table is full.
bool ShareBlock(std::int64_t physical) noexcept {
    Spin(shared_lock);
    std::size_t free_entry = SharedBlocks;
    bool done = false;
    for (std::size_t i = 0; i < SharedBlocks && !done; ++i) {
        if (shared_refs[i] > 0 && shared_block[i] == physical) {
            ++shared_refs[i];
            done = true;
        } else if (shared_refs[i] == 0 && free_entry == SharedBlocks) {
            free_entry = i;
        }
    }
    if (!done && free_entry < SharedBlocks) {
        shared_block[free_entry] = physical;
        shared_refs[free_entry] = 1;
        done = true;
    }
    shared_lock.clear(std::memory_order_release);
    return done;
}
// One slot less shows `physical`: the references left (0: the block is no longer shown).
int UnshareBlock(std::int64_t physical) noexcept {
    Spin(shared_lock);
    int left = 0;
    for (std::size_t i = 0; i < SharedBlocks; ++i)
        if (shared_refs[i] > 0 && shared_block[i] == physical) {
            left = --shared_refs[i];
            break;
        }
    shared_lock.clear(std::memory_order_release);
    return left;
}

// ---- The changes of a slot (the slot is locked) ----

// Zero -> Own: a zeroed block of its own, read-write.
bool PromoteZero(SparseRange& range, std::size_t slot) noexcept {
    DropSlotAliases(range, slot);
    const std::uintptr_t at = SlotAddress(range, slot);
#ifdef PS5_NATIVE
    std::int64_t physical = -1;
    if (!ZeroedBlock(&physical) && !ReserveBlock(&physical)) return false;
    if (!MapSlot(at, PROT_READ | PROT_WRITE, physical)) {
        (void)sceKernelReleaseDirectMemory(physical, SparseSlot);
        return false;
    }
    range.physical[slot] = physical;
#else
    if (mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_READ | PROT_WRITE) != 0) return false;
    range.physical[slot] = 0;
#endif
    guest_committed.fetch_add(SparseSlot, std::memory_order_relaxed);
    return true;
}

// Shared -> Own: the last slot to show a block takes it back read-write; the others copy it.
bool PromoteShared(SparseRange& range, std::size_t slot) noexcept {
    DropSlotAliases(range, slot);
    const std::uintptr_t at = SlotAddress(range, slot);
#ifdef PS5_NATIVE
    const std::int64_t shared = range.physical[slot];
    Spin(shared_lock);
    bool last = false;
    for (std::size_t i = 0; i < SharedBlocks; ++i)
        if (shared_refs[i] > 0 && shared_block[i] == shared) {
            last = shared_refs[i] == 1;
            if (last) shared_refs[i] = 0;
            break;
        }
    shared_lock.clear(std::memory_order_release);
    if (last) return mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_READ | PROT_WRITE) == 0;
    std::int64_t physical = -1;
    if (eden_direct_low(SparseSlot, SparseSlot, 12, &physical) != 0 && !ReserveBlock(&physical)) return false;
    const std::uintptr_t view = MapCpuView(physical, SparseSlot, PROT_READ | PROT_WRITE);
    if (view == 0) {
        (void)sceKernelReleaseDirectMemory(physical, SparseSlot);
        return false;
    }
    std::memcpy(reinterpret_cast<void*>(view), reinterpret_cast<const void*>(at), SparseSlot);
    UnmapCpuView(view, SparseSlot);
    if (!MapSlot(at, PROT_READ | PROT_WRITE, physical)) {
        (void)sceKernelReleaseDirectMemory(physical, SparseSlot);
        return false;
    }
    range.physical[slot] = physical;
    if (UnshareBlock(shared) == 0 && sceKernelReleaseDirectMemory(shared, SparseSlot) != 0) std::abort();
    guest_committed.fetch_add(SparseSlot, std::memory_order_relaxed);
    counters.copies.fetch_add(1, std::memory_order_relaxed);
    return true;
#else
    (void)range;
    (void)slot;
    (void)at;
    return false; // the host never shares
#endif
}

// Own, Shared or Fine -> Zero: the slot shows the zero block again and its memory goes back.
void ReturnSlot(SparseRange& range, std::size_t slot) noexcept {
    DropSlotAliases(range, slot);
    const std::uintptr_t at = SlotAddress(range, slot);
    const std::uint8_t state = SlotState(range, slot);
    if (state == SlotOut) {
        // No block: its place in the file goes back, and it reads as zeroes again.
#ifdef PS5_NATIVE
        if (!MapSlot(at, PROT_READ, zero_block)) std::abort();
#endif
        GiveSwapPlace(range.swap[slot]);
        range.physical[slot] = -1;
        return;
    }
    if (state == SlotProbe) {
        // Armed but never touched: an Own slot with its block, without access. Given back as the
        // Own slot it is; the mapping without access is made writable first, as the fault does.
        if (mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_READ | PROT_WRITE) != 0) std::abort();
        swap_counters.returned_armed.fetch_add(1, std::memory_order_relaxed);
    }
    if (state == SlotPacked) {
        // Its block went back when it was packed: it shows the zero block (readable) again.
#ifdef PS5_NATIVE
        if (!MapSlot(at, PROT_READ, zero_block)) std::abort();
#endif
        ForgetPacked(slot);
        range.physical[slot] = -1;
        return;
    }
    if (state == SlotFine) {
        const std::size_t kept = FineKept(range, slot);
#ifdef PS5_NATIVE
        if (!MapSlot(at, PROT_READ, zero_block)) std::abort();
        if (kept != 0 && sceKernelReleaseDirectMemory(range.physical[slot], kept * FinePage) != 0) std::abort();
#else
        if (madvise(reinterpret_cast<void*>(at), SparseSlot, MADV_DONTNEED) != 0) std::abort();
#endif
        guest_committed.fetch_sub(kept * FinePage, std::memory_order_relaxed);
        range.fine[slot] &= 0xf00;
        range.physical[slot] = -1;
        return;
    }
#ifdef PS5_NATIVE
    if (!MapSlot(at, PROT_READ, zero_block)) std::abort();
    if (state == SlotShared) {
        if (UnshareBlock(range.physical[slot]) == 0) {
            if (sceKernelReleaseDirectMemory(range.physical[slot], SparseSlot) != 0) std::abort();
            guest_committed.fetch_sub(SparseSlot, std::memory_order_relaxed);
        }
    } else {
        if (sceKernelReleaseDirectMemory(range.physical[slot], SparseSlot) != 0) std::abort();
        guest_committed.fetch_sub(SparseSlot, std::memory_order_relaxed);
    }
#else
    (void)state;
    if (madvise(reinterpret_cast<void*>(at), SparseSlot, MADV_DONTNEED) != 0 ||
        mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_READ) != 0)
        std::abort();
    guest_committed.fetch_sub(SparseSlot, std::memory_order_relaxed);
#endif
    range.physical[slot] = -1;
}

#ifdef PS5_NATIVE
// The last resort of a fault that found no block: the Eden heap's empty arenas and spaces go back
// to the pool. The performance thread trims every 2 s while the pool is short (TrimHeap); a writer
// in its fault handler cannot wait that long (on 2026-11-15 OoT gave up on a block while the heap
// held 1948 MiB with 1218 in use). Throttled once per 250 ms process-wide: the faulting threads are
// the ones the GPU waits on. Reentrancy-guarded: eden_heap_trim retires arenas under the heap's own
// locks, and a heap growth inside it must not come back here.
std::size_t TrimHeapForFault() noexcept {
    if (&eden_heap_trim == nullptr) return 0;
    timespec now{};
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0;
    const std::uint64_t ns = static_cast<std::uint64_t>(now.tv_sec) * 1'000'000'000ull +
                             static_cast<std::uint64_t>(now.tv_nsec);
    static std::atomic<std::uint64_t> last_ns{0};
    std::uint64_t last = last_ns.load(std::memory_order_relaxed);
    if (last != 0 && ns - last < 250'000'000ull) return 0;
    if (!last_ns.compare_exchange_strong(last, ns, std::memory_order_acq_rel)) return 0;
    static thread_local bool inside = false;
    if (inside) return 0;
    inside = true;
    const std::size_t freed = eden_heap_trim();
    inside = false;
    if (freed != 0) {
        static std::atomic<unsigned> notes{0};
        if (notes.fetch_add(1, std::memory_order_relaxed) < 8) {
            char line[112];
            const int length = std::snprintf(line, sizeof(line),
                                             "EDEN_CRAM the Eden heap gave back %zu MiB for a guest write\n",
                                             freed >> 20);
            if (length > 0) Note(line);
        }
    }
    return freed;
}
#endif

// Writers a fault is being served for, not only those in WaitForBlock: the whole time a write fault
// is handled. The graphics caches read it (eden_guest_memory_wanted), and GiveBackIdleBlocks and
// TakeHostReserveBlock keep the emergency blocks for it while it is nonzero. Registering here,
// before the first promotion tries to recover, is what stops that recovery from giving the reserve
// away to the host heap just before the writer waits on it.
std::atomic<unsigned> memory_waiters{0};

// A fault that found no block. Packing cold slots is what frees guest blocks; the driver's unused
// memory and the Eden heap's empty arenas/spaces may pay as well. The request covers every writer
// being served, so one pass frees enough for the whole burst rather than a block per call.
// Throttled per thread, so the fault handler's 20 ms retry loop does not turn into a pack storm.
bool RecoverForFault() noexcept {
    static thread_local std::uint64_t last_ns = 0;
    timespec now{};
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return false;
    const std::uint64_t ns = static_cast<std::uint64_t>(now.tv_sec) * 1'000'000'000ull +
                             static_cast<std::uint64_t>(now.tv_nsec);
    if (last_ns != 0 && ns - last_ns < 100'000'000ull) return false;
    last_ns = ns;
    const unsigned waiters = memory_waiters.load(std::memory_order_acquire);
    const std::size_t ask = SparseSlot * (1 + std::min<unsigned>(waiters, 32));
    std::size_t recovered = eden_memory_recover_try(ask);
#ifdef PS5_NATIVE
    if (recovered == 0) recovered = TrimHeapForFault();
#endif
    return recovered != 0;
}

void NoteUnpackWait() noexcept {
    static std::atomic<unsigned> notes{0};
    if (notes.fetch_add(1, std::memory_order_relaxed) < 24) {
        Report("EDEN_CRAM_UNPACK_WAIT a written packed slot found no block: packing cold slots and waiting\n");
    }
}

// Packed -> Own: a block of its own, read-write, with the slot's data unpacked into it.
bool UnpackSlot(SparseRange& range, std::size_t slot) noexcept {
#ifdef PS5_NATIVE
    PackedSlot* const table = packed_slots.load(std::memory_order_acquire);
    if (table == nullptr || table[slot].data == nullptr) std::abort();
    const std::uintptr_t at = SlotAddress(range, slot);
    std::int64_t physical = -1;
    if (eden_direct_low(SparseSlot, SparseSlot, 12, &physical) != 0 && !UnpackBlock(&physical) &&
        !ReserveBlock(&physical)) {
        // Nothing idle: have cold slots packed, the driver's unused memory and the Eden heap's empty
        // arenas given back, then look again. The fault path repeats this every 100 ms for up to 2 s
        // (WaitForBlock).
        RecoverForFault();
        if (eden_direct_low(SparseSlot, SparseSlot, 12, &physical) != 0 && !UnpackBlock(&physical) &&
            !ReserveBlock(&physical)) {
            NoteUnpackWait();
            return false;
        }
    }
    // Unpacked in a view of its own and only then mapped at the slot: mapped first, a reader that
    // had not faulted yet would see the block half unpacked. The view must be ours and backed
    // before LZ4 writes to it; MapCpuView maps at a free view of the range reserved for them, and
    // returns 0 (nothing written) when it cannot.
    const std::uintptr_t view = MapCpuView(physical, SparseSlot, PROT_READ | PROT_WRITE);
    if (view == 0) {
        (void)sceKernelReleaseDirectMemory(physical, SparseSlot);
        return false;
    }
    const std::uint32_t stored = table[slot].size;
    const std::uint32_t bytes = PackedStoredBytes(stored);
    bool unpacked = false;
#ifdef EDEN_SWAP_ZSTD
    if (stored & PackedZstdBit) {
        unpacked = ZstdUnpack(table[slot].data, bytes, reinterpret_cast<void*>(view));
    } else
#endif
    {
        const int out = LZ4_decompress_safe(static_cast<const char*>(table[slot].data),
                                            reinterpret_cast<char*>(view), static_cast<int>(bytes),
                                            static_cast<int>(SparseSlot));
        unpacked = out == static_cast<int>(SparseSlot);
    }
    if (!unpacked) std::abort(); // the guest's data is gone: never carry on
    if (pack_kept != nullptr && pack_kept[slot] >= 0) {
        const std::uintptr_t kept_view = MapCpuView(pack_kept[slot], SparseSlot, PROT_READ);
        if (kept_view != 0) {
            pack_keep_compared.fetch_add(1, std::memory_order_relaxed);
            const auto* a = reinterpret_cast<const std::uint64_t*>(view);
            const auto* b = reinterpret_cast<const std::uint64_t*>(kept_view);
            for (std::size_t i = 0; i < SparseSlot / 8; ++i)
                if (a[i] != b[i]) {
                    if (pack_keep_differed.fetch_add(1, std::memory_order_relaxed) < 8) {
                        char line[200];
                        const int n = std::snprintf(line, sizeof(line),
                            "EDEN_CRAM_PACK_KEEP slot %zu changed while packed: word %zu unpacked %#llx kept %#llx\n",
                            slot, i, (unsigned long long)a[i], (unsigned long long)b[i]);
                        if (n > 0) Note(line);
                    }
                    break;
                }
            UnmapCpuView(kept_view, SparseSlot);
        }
        (void)sceKernelReleaseDirectMemory(pack_kept[slot], SparseSlot);
        pack_kept[slot] = -1;
    }
    UnmapCpuView(view, SparseSlot);
    if (!MapSlot(at, PROT_READ | PROT_WRITE, physical)) {
        (void)sceKernelReleaseDirectMemory(physical, SparseSlot);
        return false;
    }
    range.physical[slot] = physical;
    ForgetPacked(slot);
    guest_committed.fetch_add(SparseSlot, std::memory_order_relaxed);
    unpacks.fetch_add(1, std::memory_order_relaxed);
    return true;
#else
    (void)range;
    (void)slot;
    return false; // the host never packs
#endif
}

bool AllZero(std::uintptr_t at) noexcept {
    const auto* word = reinterpret_cast<const volatile std::uint64_t*>(at);
    for (std::size_t i = 0; i < SparseSlot / 8; i += 8) {
        if (word[i] | word[i + 1] | word[i + 2] | word[i + 3] | word[i + 4] | word[i + 5] | word[i + 6] | word[i + 7])
            return false;
    }
    return true;
}

bool PageZero(std::uintptr_t at) noexcept {
    const auto* word = reinterpret_cast<const volatile std::uint64_t*>(at);
    for (std::size_t i = 0; i < FinePage / 8; i += 8)
        if (word[i] | word[i + 1] | word[i + 2] | word[i + 3] | word[i + 4] | word[i + 5] | word[i + 6] | word[i + 7])
            return false;
    return true;
}

// Fine -> Own: a written Fine slot becomes one block of 2 MiB again (a copy of what it shows), and
// stays whole for the rest of the session.
bool PromoteFine(SparseRange& range, std::size_t slot) noexcept {
    DropSlotAliases(range, slot);
    const std::uintptr_t at = SlotAddress(range, slot);
    const std::size_t kept = FineKept(range, slot);
#ifdef PS5_NATIVE
    std::int64_t physical = -1;
    if (eden_direct_low(SparseSlot, SparseSlot, 12, &physical) != 0 && !ReserveBlock(&physical)) return false;
    const std::uintptr_t view = MapCpuView(physical, SparseSlot, PROT_READ | PROT_WRITE);
    if (view == 0) {
        (void)sceKernelReleaseDirectMemory(physical, SparseSlot);
        return false;
    }
    std::memcpy(reinterpret_cast<void*>(view), reinterpret_cast<const void*>(at), SparseSlot);
    UnmapCpuView(view, SparseSlot);
    if (!MapSlot(at, PROT_READ | PROT_WRITE, physical)) {
        (void)sceKernelReleaseDirectMemory(physical, SparseSlot);
        return false;
    }
    if (kept != 0 && sceKernelReleaseDirectMemory(range.physical[slot], kept * FinePage) != 0) std::abort();
    range.physical[slot] = physical;
#else
    if (mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_READ | PROT_WRITE) != 0) return false;
#endif
    guest_committed.fetch_add(SparseSlot - kept * FinePage, std::memory_order_relaxed);
    const unsigned times = std::min<unsigned>(((range.fine[slot] >> 8) & 0xf) + 1, FineBackoffMax);
    range.fine[slot] = static_cast<std::uint16_t>(times << 8);
    range.fine_after[slot] = static_cast<std::uint32_t>(counters.passes.load(std::memory_order_relaxed)) + (1u << times);
    counters.unfined.fetch_add(1, std::memory_order_relaxed);
    return true;
}

#ifdef PS5_NATIVE
bool MapRun(std::uintptr_t at, std::size_t bytes, std::int64_t physical) noexcept {
    void* address = reinterpret_cast<void*>(at);
    return sceKernelMapDirectMemory(&address, bytes, PROT_READ, MAP_FIXED, physical, FinePage) == 0 &&
           address == reinterpret_cast<void*>(at);
}
#endif

// Own -> Fine (the slot is locked and fenced, and has `zero` marked in a mask of its pages): its other
// pages go into a compact block, and the slot is mapped again page run by page run, read-only.
// The 2 MiB mapping is taken away whole first (a large mapping is never split); a thread that
// touches the slot meanwhile faults, waits for the lock and repeats its access.
bool MakeFine(SparseRange& range, std::size_t slot, const bool (&zero)[FinePages], std::size_t kept) noexcept {
    const std::uintptr_t at = SlotAddress(range, slot);
#ifdef PS5_NATIVE
    std::int64_t compact = -1;
    if (eden_direct_low(kept * FinePage, FinePage, 12, &compact) != 0) return false;
    const std::uintptr_t view =
        MapCpuView(compact, kept * FinePage, PROT_READ | PROT_WRITE, FinePage);
    if (view == 0) {
        (void)sceKernelReleaseDirectMemory(compact, kept * FinePage);
        return false;
    }
    std::size_t next = 0;
    for (std::size_t page = 0; page < FinePages; ++page)
        if (!zero[page]) {
            std::memcpy(reinterpret_cast<std::uint8_t*>(view) + next * FinePage,
                        reinterpret_cast<const void*>(at + page * FinePage), FinePage);
            ++next;
        }
    UnmapCpuView(view, kept * FinePage);
    const std::int64_t whole = range.physical[slot];
    // The 2 MiB mapping is taken away whole first (a large mapping is never split): empty the slot
    // without a hole, so the system cannot place a non-fixed allocation in it between here and
    // the page runs below (MakeCpuViewRange). A reservation or a PROT_NONE guard faults a thread
    // that touches the slot meanwhile, as the munmap did.
    if (!VoidRange(at, SparseSlot)) {
        if (!MapSlot(at, PROT_READ, whole)) std::abort();
        (void)sceKernelReleaseDirectMemory(compact, kept * FinePage);
        return false;
    }
    bool ok = true;
    next = 0;
    for (std::size_t page = 0; page < FinePages && ok;) {
        std::size_t end = page + 1;
        while (end < FinePages && zero[end] == zero[page]) ++end;
        const std::size_t bytes = (end - page) * FinePage;
        ok = MapRun(at + page * FinePage, bytes, zero[page] ? zero_block : compact + std::int64_t(next * FinePage));
        if (!zero[page]) next += end - page;
        page = end;
    }
    if (!ok) {
        // The slot shows its block again, as it was (fenced: read-only until the caller unfences).
        if (!MapSlot(at, PROT_READ, whole)) std::abort();
        (void)sceKernelReleaseDirectMemory(compact, kept * FinePage);
        return false;
    }
    if (sceKernelReleaseDirectMemory(whole, SparseSlot) != 0) std::abort();
    range.physical[slot] = compact;
#else
    for (std::size_t page = 0; page < FinePages; ++page)
        if (zero[page] && madvise(reinterpret_cast<void*>(at + page * FinePage), FinePage, MADV_DONTNEED) != 0)
            std::abort();
#endif
    guest_committed.fetch_sub(SparseSlot - kept * FinePage, std::memory_order_relaxed);
    range.fine[slot] = static_cast<std::uint16_t>((range.fine[slot] & 0xf00) | kept);
    counters.fined.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// ---- Arming, touching, sending out and reading back (the slot is locked) ----

void WakeSwapper() noexcept; // below, once the scanner's wake is defined

// Own -> Probe, unless it was touched a moment ago. A hot slot pays one fault every SwapRearmTicks.
bool ArmSlot(SparseRange& range, std::size_t slot, bool at_once = false) noexcept {
#ifdef PS5_NATIVE
    LockSlot(range, slot);
    const std::uint8_t state = SlotState(range, slot);
    const std::uint32_t now = swap_tick.load(std::memory_order_relaxed);
    if (state != SlotOwn || (!at_once && now - range.swap[slot] < SwapRearmTicks)) {
        UnlockSlot(range, slot, state);
        return false;
    }
    DropSlotAliases(range, slot); // a window alias would reach the block past its arming
    if (mprotect(reinterpret_cast<void*>(SlotAddress(range, slot)), SparseSlot, PROT_NONE) != 0) std::abort();
    range.swap[slot] = now;
    swap_counters.armed.fetch_add(1, std::memory_order_relaxed);
    UnlockSlot(range, slot, SlotProbe);
    return true;
#else
    (void)range;
    (void)slot;
    (void)at_once;
    return false;
#endif
}

// Probe -> Own: the first touch, read or write, of an armed slot. The access repeats and succeeds.
bool TouchProbe(SparseRange& range, std::size_t slot) noexcept {
    const int saved = errno;
    const bool ok = mprotect(reinterpret_cast<void*>(SlotAddress(range, slot)), SparseSlot, PROT_READ | PROT_WRITE) == 0;
    errno = saved;
    if (!ok) return false;
    const std::uint32_t now = swap_tick.load(std::memory_order_relaxed), ticks = now - range.swap[slot];
    std::size_t bucket = 0;
    while (bucket + 1 < SwapAges && (std::uint64_t{1} << (bucket + 1)) <= ticks) ++bucket;
    swap_counters.age[bucket].fetch_add(1, std::memory_order_relaxed);
    swap_counters.touched.fetch_add(1, std::memory_order_relaxed);
    range.swap[slot] = now;
    return true;
}

// A 2 MiB block for a slot that comes back: its own slot's packed cache first, then the pool, then
// the guest's reserve. Nothing here allocates.
bool TakeSwapBlock(std::int64_t* physical) noexcept {
#ifdef PS5_NATIVE
    return UnpackBlock(physical) || eden_direct_low(SparseSlot, SparseSlot, 12, physical) == 0 ||
           ReserveBlock(physical);
#else
    (void)physical;
    return false;
#endif
}

// A fault needs a block and the pool has none: make memory in the fault handler, as the heap does.
// Ocarina (2026-11-15, 173 s) touched a slot that was out, no 2 MiB block was left, and the fault
// went unhandled (SIGSEGV in its JIT code). Cold slots are packed on this thread, the swapper is
// woken, the graphics caches see no memory free (memory_waiters), and every 100 ms the Eden heap's
// empty arenas go back. Up to 15 s, then the fault is refused and reported.
bool GetBlockForSwap(std::int64_t* physical) noexcept {
#ifdef PS5_NATIVE
    if (TakeSwapBlock(physical)) return true;
    const std::uint64_t start = NowNs();
    while (NowNs() - start < 15'000'000'000ull) {
        WakeSwapper();
        RecoverForFault();
        if (TakeSwapBlock(physical)) return true;
        const timespec pause{0, 20'000'000};
        nanosleep(&pause, nullptr);
    }
    Note("EDEN_SWAP a fault found no 2 MiB block in 15 s of making memory\n");
    return false;
#else
    (void)physical;
    return false;
#endif
}

// Probe -> Out: the slot's bytes go to the swap file and its block back to the pool. The slot is
// locked and protected read-only while it is written, so a reader waits for the lock and a writer
// faults and waits; a Probe slot touched meanwhile is made Own only once this is over (the fault
// waits for the lock and finds it Out). The mapping goes away whole and the address space is
// reserved at once, as the heap gives a block back.
bool SwapOut(SparseRange& range, std::size_t slot) noexcept {
#ifdef PS5_NATIVE
    LockSlot(range, slot);
    std::uint32_t place = 0;
    if (SlotState(range, slot) != SlotProbe || swap_fd < 0 || !TakeSwapPlace(&place)) {
        UnlockSlot(range, slot, SlotState(range, slot));
        return false;
    }
    const std::uintptr_t at = SlotAddress(range, slot);
    const std::uint64_t start = NowNs();
    if (mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_READ) != 0) std::abort();
    if (!SwapWrite(reinterpret_cast<const void*>(at), place)) {
        if (mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_NONE) != 0) std::abort();
        GiveSwapPlace(place);
        swap_counters.failures.fetch_add(1, std::memory_order_relaxed);
        UnlockSlot(range, slot, SlotProbe);
        return false;
    }
    const std::int64_t block = range.physical[slot];
    if (munmap(reinterpret_cast<void*>(at), SparseSlot) != 0) std::abort();
    void* again = reinterpret_cast<void*>(at);
    if (sceKernelReserveVirtualRange(&again, SparseSlot, MAP_FIXED, SparseSlot) != 0 || again != reinterpret_cast<void*>(at))
        std::abort();
    if (sceKernelReleaseDirectMemory(block, SparseSlot) != 0) std::abort();
    range.physical[slot] = -1;
    range.swap[slot] = place;
    guest_committed.fetch_sub(SparseSlot, std::memory_order_relaxed);
    swap_counters.out.fetch_add(1, std::memory_order_relaxed);
    swap_counters.out_ns.fetch_add(NowNs() - start, std::memory_order_relaxed);
    swapped_to_file.fetch_add(1, std::memory_order_relaxed);
    if (slot < SwapEverOutSlots)
        swap_ever_out[slot / 64].fetch_or(std::uint64_t{1} << (slot % 64), std::memory_order_relaxed);
    swap_given_back.fetch_add(SparseSlot, std::memory_order_relaxed);
    UnlockSlot(range, slot, SlotOut);
    return true;
#else
    (void)range;
    (void)slot;
    return false;
#endif
}

// Out -> Own, in a fault handler: a block, its bytes from the file through a view of their own, and
// the slot shows them in one step. Nothing here allocates. The place in the file goes back at once:
// the slot can be written now, and a later trip writes it again.
bool SwapIn(SparseRange& range, std::size_t slot) noexcept {
#ifdef PS5_NATIVE
    const int saved = errno;
    const std::uint64_t start = NowNs();
    const std::uint32_t place = range.swap[slot];
    std::int64_t physical = -1;
    if (!GetBlockForSwap(&physical)) {
        errno = saved;
        return false;
    }
    const std::uintptr_t view = MapCpuView(physical, SparseSlot, PROT_READ | PROT_WRITE);
    if (view == 0) {
        (void)sceKernelReleaseDirectMemory(physical, SparseSlot);
        errno = saved;
        return false;
    }
    const bool read = SwapRead(reinterpret_cast<void*>(view), place);
    UnmapCpuView(view, SparseSlot);
    if (!read || !MapSlot(SlotAddress(range, slot), PROT_READ | PROT_WRITE, physical)) {
        (void)sceKernelReleaseDirectMemory(physical, SparseSlot);
        swap_counters.failures.fetch_add(1, std::memory_order_relaxed);
        Note(read ? "EDEN_SWAP a slot could not be mapped again\n" : "EDEN_SWAP a slot could not be read back from the file\n");
        errno = saved;
        return false;
    }
    range.physical[slot] = physical;
    range.swap[slot] = swap_tick.load(std::memory_order_relaxed);
    GiveSwapPlace(place);
    guest_committed.fetch_add(SparseSlot, std::memory_order_relaxed);
    const std::uint64_t took = NowNs() - start;
    swap_counters.in.fetch_add(1, std::memory_order_relaxed);
    swap_counters.in_ns.fetch_add(took, std::memory_order_relaxed);
    if (took > swap_counters.in_most_ns.load(std::memory_order_relaxed))
        swap_counters.in_most_ns.store(took, std::memory_order_relaxed);
    errno = saved;
    return true;
#else
    (void)range;
    (void)slot;
    return false;
#endif
}

// CommitSparsePage for a range with slot states.
bool ServiceCommit(SparseRange& range, std::size_t slot) noexcept {
    LockSlot(range, slot);
    const std::uint8_t state = SlotState(range, slot);
    const bool ok = state == SlotZero ? PromoteZero(range, slot)
                  : state == SlotFine ? PromoteFine(range, slot)
                  : state == SlotPacked ? UnpackSlot(range, slot)
                  : state == SlotProbe ? TouchProbe(range, slot)
                  : state == SlotOut ? SwapIn(range, slot) : true;
    UnlockSlot(range, slot,
               ok && (state == SlotZero || state == SlotFine || state == SlotPacked || state == SlotProbe || state == SlotOut)
                   ? SlotOwn
                   : state);
    return ok;
}

// ---- The fault ----
SparseRange* WriteRangeOf(std::uintptr_t address) noexcept {
    for (SparseRange* range : {service_range.load(std::memory_order_acquire), probe_range.load(std::memory_order_acquire)}) {
        if (range == nullptr || range->state == nullptr) continue;
        const auto begin = range->begin.load(std::memory_order_acquire);
        if (begin != 0 && address >= begin && address < range->end) return range;
    }
    return nullptr;
}
std::atomic<std::uintptr_t> probe_expected{0}; // the self-test's write, whatever si_addr says
std::atomic<bool> probe_mismatch{false};
// The last write the service refused, for the crash report: the address, then why (1: it faulted
// again and again with nothing to change, 2: no block was left for its slot) and the slot's state.
std::atomic<std::uintptr_t> refused_address{0};
std::atomic<unsigned> refused_detail{0}; // why | state << 8
void Refused(std::uintptr_t address, unsigned why, std::uint8_t state) noexcept {
    refused_detail.store(why | unsigned{state} << 8, std::memory_order_relaxed);
    refused_address.store(address, std::memory_order_release);
}
} // namespace

// The last write the service refused (headless/crash_report.cpp): its address, 0 for none.
extern "C" std::uintptr_t eden_guest_write_refused(unsigned* why, unsigned* state) noexcept {
    const std::uintptr_t address = refused_address.load(std::memory_order_acquire);
    const unsigned detail = refused_detail.load(std::memory_order_relaxed);
    *why = detail & 0xff;
    *state = detail >> 8;
    return address;
}

#ifdef PS5_NATIVE
extern "C" const char __eden_text_start[] __attribute__((weak));
extern "C" const char __eden_text_end[] __attribute__((weak));
#endif

namespace {
// Who writes the slots that get promoted. A fault handler (ServiceSignal) reads the interrupted
// instruction's address and the first return address of the app's code above the stack pointer:
// a write by the emulated CPU has the JIT's entry point there, a write by the emulator itself the
// function that called the copy. The promotions are counted by that place (ServiceReport).
constexpr std::size_t WriterPcs = 4;
struct WriterSite {
    std::atomic<std::uintptr_t> key{0};
    std::uintptr_t pcs[WriterPcs]{};
    std::atomic<std::uint64_t> count{0};
};
constexpr std::size_t WriterSites = 512;
WriterSite writer_sites[WriterSites];
std::atomic<std::uint64_t> promotions_in_app_code{0}, promotions_elsewhere{0};
thread_local std::uintptr_t fault_pcs[WriterPcs] = {};
thread_local bool fault_in_app_code = false;

void NoteWriter() noexcept {
    (fault_in_app_code ? promotions_in_app_code : promotions_elsewhere).fetch_add(1, std::memory_order_relaxed);
    std::uintptr_t key = 0xCBF29CE484222325ull;
    for (const std::uintptr_t pc : fault_pcs) key = (key ^ pc) * 0x100000001B3ull;
    key |= 1;
    std::size_t at = static_cast<std::size_t>((key >> 12) % WriterSites);
    for (std::size_t probe = 0; probe < 16; ++probe, at = (at + 1) % WriterSites) {
        std::uintptr_t expected = 0;
        if (writer_sites[at].key.compare_exchange_strong(expected, key, std::memory_order_acq_rel)) {
            for (std::size_t i = 0; i < WriterPcs; ++i) writer_sites[at].pcs[i] = fault_pcs[i];
            writer_sites[at].count.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (expected == key) {
            writer_sites[at].count.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
}
} // namespace

namespace {
// How many waits ended with a block (WaitForBlock). The writers still being served are counted in
// memory_waiters, above.
std::atomic<std::uint64_t> memory_waits_won{0}, memory_waits_lost{0};

// No block was left for a written slot. Most of what the pool gives out is held by the graphics
// caches, which free what is not drawn once they see memory short (headless/performance.cpp reads
// eden_guest_memory_wanted, and while a writer waits the caches see no free memory at all). The
// game's own memory comes first: a large open-world game wrote 1.8 GB of new memory within 10 s
// after 23 minutes of play, faster than the caches free at the end of each frame, and the write
// that found no block ended the session. So the writer waits here, in its fault handler, and tries
// again every 20 ms for up to 2 s while the caches give memory back. Only then is it refused.
template <typename Promote>
bool WaitForBlock(Promote&& promote) noexcept {
    bool ok = false;
    for (unsigned tries = 0; tries < 100 && !ok; ++tries) {
        // Every 100 ms, ask again for what the other owners can give: a fault that is not on a
        // packed slot would otherwise wait with nothing packing or trimming.
        if (tries % 5 == 0) RecoverForFault();
        const timespec pause{0, 20'000'000};
        nanosleep(&pause, nullptr);
        ok = promote();
    }
    (ok ? memory_waits_won : memory_waits_lost).fetch_add(1, std::memory_order_relaxed);
    Note(ok ? "EDEN_CRAM a written slot waited for the graphics caches to give memory back, and got a block\n"
            : "EDEN_CRAM a written slot waited 2 s for memory and found no block\n");
    return ok;
}
} // namespace

// How many writes of the guest's memory are waiting for a block now (headless/performance.cpp):
// while any is, the graphics caches are told no memory is free.
// A page of the guest's memory the GPU's caches now hold (+1) or no longer hold (-1), from the core's
// device memory manager. Counted per slot; packing skips slots with any (PackSlot).
extern "C" void eden_guest_gpu_cached(const void* host, int delta) noexcept {
    SparseRange* const range = guest_range.load(std::memory_order_acquire);
    if (range == nullptr) return;
    const std::uintptr_t begin = range->begin.load(std::memory_order_relaxed);
    const auto at = reinterpret_cast<std::uintptr_t>(host);
    if (at < begin || at >= range->end) return;
    auto* table = gpu_cached_table.load(std::memory_order_acquire);
    if (table == nullptr) {
        const std::size_t slots = (range->end - begin) / SparseSlot;
        auto* made = new (std::nothrow) std::atomic<std::int32_t>[slots]();
        if (made == nullptr) return;
        gpu_cached_slots = slots;
        if (gpu_cached_table.compare_exchange_strong(table, made, std::memory_order_acq_rel)) table = made;
        else delete[] made;
    }
    const std::size_t slot = (at - begin) / SparseSlot;
    if (slot < gpu_cached_slots) table[slot].fetch_add(delta, std::memory_order_acq_rel);
}

extern "C" unsigned eden_guest_memory_wanted(void) noexcept {
    return memory_waiters.load(std::memory_order_acquire);
}

// Last resort for a host heap piece, after the direct pool refused even 2 MiB.
// Transfer ownership under the reserve lock: never release then race the keeper
// to reacquire the same memory. The host's committed-block table owns/frees it.
// Leave at least half the configured reserve for guest write faults, and give
// waiting guest writers priority. This does not change the guest's own fallback.
bool TakeHostReserveBlock(std::int64_t* physical) noexcept {
#ifdef PS5_NATIVE
    const char* enabled = std::getenv("EDEN_HOST_RESERVE_TRANSFER");
    if (!physical || !enabled || std::strcmp(enabled, "1") != 0) return false;
    Spin(reserve_lock);
    const auto count = reserve_count.load(std::memory_order_relaxed);
    const bool allowed = service_range.load(std::memory_order_acquire) != nullptr &&
        memory_waiters.load(std::memory_order_acquire) == 0 && count > ReserveTarget() / 2;
    if (allowed) {
        *physical = reserve[count - 1];
        reserve_count.store(count - 1, std::memory_order_relaxed);
    }
    reserve_lock.clear(std::memory_order_release);
    if (allowed) counters.host_reserve_used.fetch_add(1, std::memory_order_relaxed);
    return allowed;
#else
    (void)physical;
    return false;
#endif
}

// The guest's headroom (the reserve): what it holds and what it is to hold, in bytes; both 0 while
// the service does not run or keeps no reserve (the host). headless/performance.cpp counts what is
// not held yet as taken, as all of it was before.
extern "C" void eden_guest_memory_reserve(std::size_t* held, std::size_t* target) noexcept {
#ifdef PS5_NATIVE
    const bool running = service_range.load(std::memory_order_acquire) != nullptr;
    *held = running ? reserve_count.load(std::memory_order_relaxed) * SparseSlot : 0;
    *target = running ? ReserveTarget() * SparseSlot : 0;
#else
    *held = 0;
    *target = 0;
#endif
}

// How the waits ended so far, and the writes refused for want of a block (headless/performance.cpp
// ReportMemory prints them beside what holds the pool).
extern "C" void eden_guest_memory_waits(unsigned long long* won, unsigned long long* lost,
                                        unsigned long long* refused) noexcept {
    *won = memory_waits_won.load(std::memory_order_relaxed);
    *lost = memory_waits_lost.load(std::memory_order_relaxed);
    *refused = counters.failures.load(std::memory_order_relaxed);
}

// A write fault at `address`: true when it was a write to a slot of the guest's memory that is
// now writable (the faulting instruction repeats), false for anything else.
bool HandleGuestWriteFault(void* fault) noexcept {
    auto address = reinterpret_cast<std::uintptr_t>(fault);
    if (const auto expected = probe_expected.load(std::memory_order_acquire); expected != 0 && address != expected) {
        probe_mismatch.store(true, std::memory_order_relaxed);
        address = expected;
    }
    SparseRange* const range = WriteRangeOf(address);
    if (range == nullptr) return false;
    // A fault being served is a writer waiting for a block even before its first promotion fails:
    // packing, GiveBackIdleBlocks and the graphics caches must not give the emergency blocks away
    // while it runs.
    struct WaiterRegistration {
        WaiterRegistration() noexcept { memory_waiters.fetch_add(1, std::memory_order_acq_rel); }
        ~WaiterRegistration() { memory_waiters.fetch_sub(1, std::memory_order_acq_rel); }
    } const waiter;
    const std::size_t slot = (address - range->begin.load(std::memory_order_relaxed)) / SparseSlot;
    // The same address faulting again and again with nothing to change is not a slot to promote
    // but a real fault. A promotion starts the count again (below).
    static thread_local std::uintptr_t last = 0;
    static thread_local unsigned repeats = 0;
    repeats = address == last ? repeats + 1 : 1;
    last = address;
    if (repeats > 16) {
        repeats = 0;
        Refused(address, 1, SlotState(*range, slot));
        Note("EDEN_CRAM refused a write: 17 faults in a row at one address, its slot unchanged\n");
        return false;
    }
    counters.faults.fetch_add(1, std::memory_order_relaxed);
    const bool waited = LockSlot(*range, slot, true);
    const std::uint8_t state = SlotState(*range, slot);
    if (waited) {
        // Someone changed the slot meanwhile (or is mapping it again): the access may now succeed,
        // and a read must never make a Fine slot whole. It faults again if it was a write.
        UnlockSlot(*range, slot, state);
        return true;
    }
    const auto promote = [&]() noexcept {
        bool done = true;
        if (state == SlotFine) {
            done = PromoteFine(*range, slot);
        } else if (state == SlotZero) {
            done = PromoteZero(*range, slot);
            if (done) {
                counters.promotions.fetch_add(1, std::memory_order_relaxed);
                NoteWriter();
            }
        } else if (state == SlotShared) {
            done = PromoteShared(*range, slot);
        } else if (state == SlotPacked) {
            done = UnpackSlot(*range, slot);
            if (done) CountUnpacker(slot);
        } else if (state == SlotProbe) {
            done = TouchProbe(*range, slot);
        } else if (state == SlotOut) {
            done = SwapIn(*range, slot);
        }
        return done;
    };
    bool ok = promote();
    if (!ok) {
        // Nothing idle for the first try: have the other owners give memory back now (cold slots
        // packed, RADV trimmed, the Eden heap's empty arenas), then look again before waiting.
        RecoverForFault();
        ok = promote();
    }
    if (!ok) ok = WaitForBlock(promote);
    UnlockSlot(*range, slot, ok ? SlotOwn : state);
    if (!ok) {
        counters.failures.fetch_add(1, std::memory_order_relaxed);
        Refused(address, 2, state);
        char line[256];
        const int length = std::snprintf(line, sizeof(line),
            "EDEN_GUEST_OOM the guest's memory found no block (fault slot %zu, state %u): "
            "EDEN_CRAM out of memory, waiters=%u reserve=%zuMiB unpack_keep=%zuMiB packed=%zuMiB stored=%zuMiB\n",
            slot, unsigned(state), memory_waiters.load(std::memory_order_relaxed),
            reserve_count.load(std::memory_order_relaxed) * (SparseSlot >> 20),
            unpack_keep_count.load(std::memory_order_relaxed) * (SparseSlot >> 20),
            static_cast<std::size_t>(packed_bytes.load(std::memory_order_relaxed) >> 20),
            store_held.load(std::memory_order_relaxed) >> 20);
        if (length > 0) Note(line);
    } else if (state != SlotOwn) {
        // The slot was made writable. A game that writes one place each time its slot was given
        // back (cleared, freed or scanned) faults there again and again over a session, each time
        // with something to do: counted as repeats, the 17th such write ended the game after
        // 15 to 60 minutes (Swordpdf/ProsperoEden#246).
        repeats = 0;
    }
    return ok;
}

namespace {
struct sigaction service_previous_segv{}, service_previous_bus{};

void ServiceSignal(int signal, siginfo_t* info, void* context) {
#ifdef PS5_NATIVE
    fault_in_app_code = false;
    for (std::uintptr_t& pc : fault_pcs) pc = 0;
    if (context != nullptr) {
        const auto low = reinterpret_cast<std::uintptr_t>(__eden_text_start) + 8;
        const auto high = reinterpret_cast<std::uintptr_t>(__eden_text_end);
        const std::uintptr_t rip = Eden::Fastmem::ContextRip(context);
        fault_in_app_code = rip >= low && rip < high;
        const auto* stack = reinterpret_cast<const std::uintptr_t*>(Eden::Fastmem::ContextRsp(context));
        std::size_t found = 0;
        // Never past the page (16 KiB) that holds the stack pointer: a thread's stack can end a few
        // hundred bytes above it, and what lies after it is not mapped.
        const auto page_end = ((reinterpret_cast<std::uintptr_t>(stack) | 0x3FFFu) + 1);
        const auto room = static_cast<unsigned>((page_end - reinterpret_cast<std::uintptr_t>(stack)) / sizeof(std::uintptr_t));
        for (unsigned i = 0; i < 160 && i < room && found < WriterPcs; ++i) {
            if (stack[i] >= low && stack[i] < high && (found == 0 || fault_pcs[found - 1] != stack[i])) {
                fault_pcs[found++] = stack[i];
            }
        }
    }
#endif
    if (HandleGuestWriteFault(info ? info->si_addr : nullptr)) return;
    const struct sigaction& previous = signal == SIGSEGV ? service_previous_segv : service_previous_bus;
    if (previous.sa_flags & SA_SIGINFO) {
        previous.sa_sigaction(signal, info, context);
    } else if (previous.sa_handler == SIG_DFL) {
        std::signal(signal, SIG_DFL); // the faulting instruction repeats and ends the process
    } else if (previous.sa_handler != SIG_IGN) {
        previous.sa_handler(signal);
    }
}

bool InstallServiceSignal() noexcept {
    static std::once_flag once;
    static bool installed = false;
    std::call_once(once, [] {
        struct sigaction action{};
        action.sa_sigaction = ServiceSignal;
        action.sa_flags = SA_SIGINFO | SA_RESTART;
        sigemptyset(&action.sa_mask);
        installed = sigaction(SIGSEGV, &action, &service_previous_segv) == 0 &&
                    sigaction(SIGBUS, &action, &service_previous_bus) == 0;
    });
    return installed;
}

// Gives a range slot states (a guest range, or the self-test's).
bool AddStates(SparseRange& range) noexcept {
    const std::size_t slots = (range.end - range.begin.load(std::memory_order_relaxed)) / SparseSlot;
    auto* state = new (std::nothrow) std::atomic<std::uint8_t>[slots];
    auto* fine = state ? new (std::nothrow) std::uint16_t[slots]() : nullptr;
    auto* fine_after = fine ? new (std::nothrow) std::uint32_t[slots]() : nullptr;
    auto* aliases = fine_after && range.guest ? new (std::nothrow) std::uintptr_t[slots * AliasesPerSlot]() : nullptr;
    auto* swap = fine_after ? new (std::nothrow) std::uint32_t[slots]() : nullptr;
    if (fine_after == nullptr || swap == nullptr || (range.guest && aliases == nullptr)) {
        delete[] state;
        delete[] fine;
        delete[] fine_after;
        delete[] swap;
        delete[] aliases;
        return false;
    }
    range.fine = fine;
    range.fine_after = fine_after;
    range.aliases = aliases;
    range.swap = swap;
    for (std::size_t slot = 0; slot < slots; ++slot)
        state[slot].store(range.owned[slot].load(std::memory_order_relaxed) ? SlotOwn : SlotZero, std::memory_order_relaxed);
    range.state = state;
    return true;
}

// ---- The scanner ----
std::thread scanner;
std::mutex scanner_mutex;
std::condition_variable scanner_wake;
bool scanner_stop = false;
#ifdef PS5_NATIVE
// The reserve kept full: every 100 ms, so a burst of writes that used it finds it whole again as
// soon as the caches have given memory back (the scanner only passes every 20 s).
std::thread keeper;

void ReserveKeeper() {
    std::unique_lock lock{scanner_mutex};
    while (!scanner_stop) {
        scanner_wake.wait_for(lock, std::chrono::milliseconds(100), [] { return scanner_stop; });
        if (scanner_stop) break;
        lock.unlock();
        FillReserve();
        lock.lock();
    }
}
#endif

std::uint64_t Mix(std::uint64_t hash, std::uint64_t word) noexcept {
    hash ^= word + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    return hash * 0xff51afd7ed558ccdull;
}
std::uint64_t SampleHash(std::uintptr_t at) noexcept {
    const auto* word = reinterpret_cast<const volatile std::uint64_t*>(at);
    std::uint64_t hash = 0;
    for (std::size_t i = 0; i < SparseSlot / 8; i += SparseSlot / 8 / 64) hash = Mix(hash, word[i] ^ word[i + 1]);
    return hash;
}
std::uint64_t FullHash(std::uintptr_t at, std::size_t bytes) noexcept {
    const auto* word = reinterpret_cast<const volatile std::uint64_t*>(at);
    std::uint64_t hash = 0;
    for (std::size_t i = 0; i < bytes / 8; ++i) hash = Mix(hash, word[i]);
    return hash;
}

void Fence(std::uintptr_t at) noexcept {
    // A window alias would write past the fence.
    if (SparseRange* range = guest_range.load(std::memory_order_acquire); range && range->aliases &&
        at >= range->begin.load(std::memory_order_relaxed) && at < range->end)
        DropSlotAliases(*range, (at - range->begin.load(std::memory_order_relaxed)) / SparseSlot);
    if (mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_READ) != 0) std::abort();
}
void Unfence(std::uintptr_t at) noexcept {
    if (mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_READ | PROT_WRITE) != 0) std::abort();
}

// An Own slot that holds only zeroes goes back. Fenced while it is looked at again.
bool ScanZero(SparseRange& range, std::size_t slot) noexcept {
    LockSlot(range, slot);
    if (SlotState(range, slot) != SlotOwn) {
        UnlockSlot(range, slot, SlotState(range, slot));
        return false;
    }
    const std::uintptr_t at = SlotAddress(range, slot);
    Fence(at);
    const bool zero = AllZero(at);
    if (zero) ReturnSlot(range, slot);
    else Unfence(at);
    UnlockSlot(range, slot, zero ? SlotZero : SlotOwn);
    return zero;
}

// An Own slot with many pages of zeroes becomes Fine. A racy count first, then the count again fenced.
bool ScanFine(SparseRange& range, std::size_t slot) noexcept {
    const std::uintptr_t at = SlotAddress(range, slot);
    if (counters.passes.load(std::memory_order_relaxed) < range.fine_after[slot]) return false;
    std::size_t zeros = 0;
    for (std::size_t page = 0; page < FinePages; ++page) zeros += PageZero(at + page * FinePage);
    if (zeros < FineMinZeroPages) return false;
    LockSlot(range, slot);
    if (SlotState(range, slot) != SlotOwn) {
        UnlockSlot(range, slot, SlotState(range, slot));
        return false;
    }
    Fence(at);
    bool zero[FinePages];
    zeros = 0;
    for (std::size_t page = 0; page < FinePages; ++page) zeros += zero[page] = PageZero(at + page * FinePage);
    bool fined = false;
    if (zeros == FinePages) {
        ReturnSlot(range, slot);
        UnlockSlot(range, slot, SlotZero);
        counters.returns_scan.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    if (zeros >= FineMinZeroPages) {
        fined = MakeFine(range, slot, zero, FinePages - zeros);
        if (!fined) counters.fine_failures.fetch_add(1, std::memory_order_relaxed);
    }
    if (!fined) Unfence(at);
    UnlockSlot(range, slot, fined ? SlotFine : SlotOwn);
    return fined;
}

// Two slots of the same bytes show one block (the console; the host only counts them).
bool ScanShare(SparseRange& range, std::size_t keep, std::size_t other) noexcept {
    const std::size_t first = std::min(keep, other), second = std::max(keep, other);
    LockSlot(range, first);
    LockSlot(range, second);
    const std::uint8_t keep_state = SlotState(range, keep), other_state = SlotState(range, other);
    bool shared = false;
    if ((keep_state == SlotOwn || keep_state == SlotShared) && other_state == SlotOwn) {
        const std::uintptr_t keep_at = SlotAddress(range, keep), other_at = SlotAddress(range, other);
        if (keep_state == SlotOwn) Fence(keep_at);
        Fence(other_at);
        const bool same = std::memcmp(reinterpret_cast<const void*>(keep_at), reinterpret_cast<const void*>(other_at),
                                      SparseSlot) == 0;
#ifdef PS5_NATIVE
        // A block becoming shared counts both slots (the first ShareBlock makes it 1, the second 2); one
        // already shared counts one more. A table that is full shares nothing.
        bool counted = false;
        if (same && ShareBlock(range.physical[keep])) {
            counted = keep_state == SlotShared || ShareBlock(range.physical[keep]);
            if (!counted) (void)UnshareBlock(range.physical[keep]);
        }
        if (counted) {
            if (!MapSlot(other_at, PROT_READ, range.physical[keep])) std::abort();
            if (sceKernelReleaseDirectMemory(range.physical[other], SparseSlot) != 0) std::abort();
            range.physical[other] = range.physical[keep];
            guest_committed.fetch_sub(SparseSlot, std::memory_order_relaxed);
            shared = true;
        }
#else
        (void)same;
#endif
        if (!shared) {
            if (keep_state == SlotOwn) Unfence(keep_at);
            Unfence(other_at);
        }
        if (same) counters.dedup.fetch_add(1, std::memory_order_relaxed);
    }
    UnlockSlot(range, second, second == keep ? (shared ? SlotShared : keep_state) : (shared ? SlotShared : other_state));
    UnlockSlot(range, first, first == keep ? (shared ? SlotShared : keep_state) : (shared ? SlotShared : other_state));
    return shared;
}

// ---- Packing (see "Packed slots" above) ----
// EDEN_CRAM_PACK=0 turns it off. While the pool's largest free block is under PackBelowLargest,
// each scan pass packs up to PackPerPass slots unwritten for ColdSeconds or more, the
// coldest first. A slot packed before waits twice as many passes each time (it came back), and
// one that does not compress to PackMaxRatio is left as it is.
// On unless EDEN_CRAM_PACK=0 (2026-11-09). It only starts once the pool's largest free block is
// below EDEN_CRAM_PACK_BELOW_MIB, so a game that fits never packs. Without PS5 Pro memory (12,224
// MiB) Mario Kart World ran out at every race load without it; with it (GPU-reachable slots
// included, below) it ran a whole race and into the next (docs/MKW-MEMORY-LEAK.md). It was off from
// 2026-11-08: sessions that packed showed impossible draws and GPU hangs, but those draws turned out
// to come without packing too (the race-load event), and the hangs with GPU threads waiting for
// memory went with the unpack reserve (EDEN_CRAM_UNPACK_KEEP_MIB).
bool PackEnabled() noexcept {
    static const bool on = [] {
        const char* value = std::getenv("EDEN_CRAM_PACK");
        return (value == nullptr || value[0] != '0') && &LZ4_compress_default != nullptr &&
               &LZ4_decompress_safe != nullptr && &LZ4_compressBound != nullptr;
    }();
    return on;
}
constexpr std::size_t PackPerPass = 256;                       // 512 MiB of slots
// EDEN_CRAM_PACK_BELOW_MIB / EDEN_CRAM_PACK_COLD_S: when packing starts (largest free block) and how long
// a slot must stay unwritten (tuning; defaults 3072 MiB and 30 s: MKW without Pro memory needs packing
// to start well before the race load, which takes ~1.5 GB at once).
std::size_t EnvOr(const char* name, std::size_t fallback) noexcept {
    const char* value = std::getenv(name);
    return value != nullptr && *value ? static_cast<std::size_t>(std::strtoull(value, nullptr, 10)) : fallback;
}
std::size_t PackBelowLargest() noexcept {
    static const std::size_t bytes = EnvOr("EDEN_CRAM_PACK_BELOW_MIB", 3072) << 20;
    return bytes;
}
constexpr double PackMaxRatio = 0.6;
// zstd is slower to unpack than LZ4, so it is kept for slots LZ4 packs too little; it is taken when
// it gives back at least this many pages (256 KiB), which is what the fork's PackOut asked.
constexpr std::size_t PackMinGainPages = 16;
std::vector<std::uint8_t> pack_times; // per slot: how often it was packed or tried (scanner, recovery)
std::atomic<bool> pack_refused{false};
std::mutex pack_mutex; // the scanner's cold pass and packing, and eden_memory_recover
thread_local bool holding_pack = false; // this thread holds pack_mutex (a heap growth inside must not wait for it)
// A RecoverMemory pack loop is running (any thread). GiveBackIdleBlocks leaves the guest's
// blocks alone while one runs: the pack may need one next.
std::atomic<unsigned> recover_active{0};
std::vector<std::size_t> recover_order; // eden_memory_recover's list, room made once (no allocation there)
std::vector<std::size_t> pack_ready;    // PackPass's, likewise
std::vector<char> pack_scratch;

#ifdef PS5_NATIVE
extern "C" [[gnu::weak]] std::int32_t sceKernelAvailableDirectMemorySize(std::int64_t, std::int64_t, std::size_t, std::int64_t*, std::size_t*);
#endif
bool PoolShortForPacking() noexcept {
#ifdef PS5_NATIVE
    const auto total = sceKernelGetDirectMemorySize();
    std::int64_t start = 0;
    std::size_t largest = 0;
    return total > 0 && &sceKernelAvailableDirectMemorySize != nullptr &&
           sceKernelAvailableDirectMemorySize(0, total, SparseSlot, &start, &largest) == 0 &&
           largest < PackBelowLargest();
#else
    return false;
#endif
}

// Own -> Packed (the scanner): fenced while it is compressed, then the zero block with no access.
bool PackSlot(SparseRange& range, std::size_t slot) noexcept {
#ifdef PS5_NATIVE
    PackedSlot* const table = packed_slots.load(std::memory_order_acquire);
    if (table == nullptr) return false;
    static const bool pack_gpu = [] {
        const char* value = std::getenv("EDEN_CRAM_PACK_GPU");
        return value == nullptr || value[0] != '0';
    }();
    if (auto* gpu_cached = gpu_cached_table.load(std::memory_order_acquire);
        !pack_gpu && gpu_cached != nullptr && slot < gpu_cached_slots &&
        gpu_cached[slot].load(std::memory_order_acquire) > 0) {
        pack_gpu_skips.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (!pack_gpu && gpu_touched != nullptr && slot < gpu_cached_slots &&
        gpu_touched[slot].load(std::memory_order_acquire) != 0) {
        pack_gpu_skips.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    LockSlot(range, slot);
    if (SlotState(range, slot) != SlotOwn) {
        UnlockSlot(range, slot, SlotState(range, slot));
        return false;
    }
    const std::uintptr_t at = SlotAddress(range, slot);
    Fence(at);
    int size = LZ4_compress_default(reinterpret_cast<const char*>(at), pack_scratch.data(),
                                    static_cast<int>(SparseSlot), static_cast<int>(pack_scratch.size()));
    bool zstd = false;
#ifdef EDEN_SWAP_ZSTD
    // LZ4 packs too little to be worth keeping this slot in memory: zstd level 1 packs Ocarina's
    // poorly-compressing slots tighter before they are sent whole to the file.
    if ((size <= 0 || size > static_cast<int>(SparseSlot * PackMaxRatio)) &&
        zstd_ready.load(std::memory_order_acquire)) {
        const std::size_t packed = ZSTD_compressCCtx(zstd_pack, pack_scratch.data(), pack_scratch.size(),
                                                     reinterpret_cast<const void*>(at), SparseSlot, 1);
        if (!ZSTD_isError(packed) && packed > 0 && packed <= SparseSlot - PackMinGainPages * FinePage) {
            size = static_cast<int>(packed);
            zstd = true;
            packed_zstd.fetch_add(1, std::memory_order_relaxed);
        }
    }
#endif
    // LZ4 must pack to PackMaxRatio; zstd only has to give back PackMinGainPages.
    const std::size_t max_size =
        zstd ? SparseSlot - PackMinGainPages * FinePage : static_cast<std::size_t>(SparseSlot * PackMaxRatio);
    if (size <= 0 || static_cast<std::size_t>(size) > max_size || !StoreStart()) {
        Unfence(at);
        UnlockSlot(range, slot, SlotOwn);
        pack_skips.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    // No access first: from here every reader faults and waits for the slot. Mapped readable even
    // for a moment, the zero block would show zeroes where the guest's data was (a pointer read
    // there made a guest thread jump to address 0).
    if (mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_NONE) != 0) std::abort();
    if (!MapSlot(at, PROT_NONE, zero_block)) {
        // The system will not map with no access: the slot stays as it was, and packing stops.
        if (mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_READ | PROT_WRITE) != 0) std::abort();
        UnlockSlot(range, slot, SlotOwn);
        pack_refused.store(true, std::memory_order_relaxed);
        Note("EDEN_CRAM_PACK off: the system refused a no-access mapping\n");
        return false;
    }
    // The slot's block, mapped nowhere now: the store may take it when the pool has no block.
    std::int64_t donate = PackKeep() && pack_kept != nullptr ? -1 : range.physical[slot];
    void* const data = StoreAlloc(static_cast<std::size_t>(size), &donate);
    if (data == nullptr) {
        // No room anywhere (the store's address space is full): the block, untouched, comes back.
        if (!MapSlot(at, PROT_READ | PROT_WRITE, range.physical[slot])) std::abort();
        UnlockSlot(range, slot, SlotOwn);
        pack_skips.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    std::memcpy(data, pack_scratch.data(), static_cast<std::size_t>(size));
    if (PackKeep() && pack_kept != nullptr) {
        pack_kept[slot] = range.physical[slot];
    } else if (donate >= 0 && !KeepForUnpack(donate) && sceKernelReleaseDirectMemory(donate, SparseSlot) != 0) {
        std::abort();
    }
    range.physical[slot] = -1;
    guest_committed.fetch_sub(SparseSlot, std::memory_order_relaxed);
    table[slot] = {data, static_cast<std::uint32_t>(size) | (zstd ? PackedZstdBit : 0)};
    packed_count.fetch_add(1, std::memory_order_relaxed);
    packed_bytes.fetch_add(static_cast<std::uint64_t>(size), std::memory_order_relaxed);
    packs.fetch_add(1, std::memory_order_relaxed);
    UnlockSlot(range, slot, SlotPacked);
    return true;
#else
    (void)range;
    (void)slot;
    return false;
#endif
}

// ---- Research: how much of the held memory is unwritten for a while, and how it compresses ----
// EDEN_CRAM_COLD_REPORT=1: every pass hashes every 64th word of each Own slot; a slot whose hash
// stays the same is counted as unwritten for that many passes (reads are not seen). Up to
// ColdSamples slots unwritten for ColdSeconds are compressed with LZ4 per pass, to estimate
// what compressing them would give back. Measurement only: nothing changes in the slots.
bool ColdReport() noexcept {
    static const bool on = [] {
        const char* value = std::getenv("EDEN_CRAM_COLD_REPORT");
        return value != nullptr && value[0] == '1';
    }();
    return on;
}
constexpr std::size_t ColdSamples = 24;
std::size_t ColdSecondsNow() noexcept {
    static const std::size_t seconds = EnvOr("EDEN_CRAM_PACK_COLD_S", 30);
    return seconds;
}
#define ColdSeconds ColdSecondsNow()
std::vector<std::uint64_t> cold_hash;
std::vector<std::uint32_t> cold_since; // per Own slot: when its hash last changed (s, steady clock)
std::uint32_t ColdNow() noexcept {
    return static_cast<std::uint32_t>(std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
std::size_t cold_rotor = 0;
std::atomic<std::uint64_t> cold_1{0}, cold_3{0}, cold_6{0}, cold_own{0}, cold_zero{0}, cold_zero_old{0};
// Diagnostic (MKW leak): every 64th word of the slot is zero.
bool ColdLooksZero(std::uintptr_t at) noexcept {
    const auto* word = reinterpret_cast<const volatile std::uint64_t*>(at);
    for (std::size_t i = 0; i < SparseSlot / 8; i += 64)
        if (word[i] != 0) return false;
    return true;
}
std::atomic<std::uint64_t> cold_sampled_bytes{0}, cold_compressed_bytes{0}, cold_compress_ns{0};

std::uint64_t ColdHash(std::uintptr_t at) noexcept {
    const auto* word = reinterpret_cast<const volatile std::uint64_t*>(at);
    std::uint64_t hash = 0;
    for (std::size_t i = 0; i < SparseSlot / 8; i += 64) hash = Mix(hash, word[i]);
    return hash;
}

void PackPass(SparseRange& range, std::size_t slots, std::vector<std::size_t>& candidates) noexcept;

// Diagnostic: the zero block (every untouched slot shows it, and the sparse tables' unused parts) must
// stay zero. Written through a mapping that should not allow it, every untouched slot of the game
// would read that garbage. Checked each pass through a read-only view of its own, mapped from
// the view range and given back at once: a mapping the kernel places (flags 0, a hint) can sit
// inside a range the app reserved at the same hint, and a MAP_FIXED commit or an unmap of that
// range takes the view away under the check (2026-11-14: the check read 0x1180000000, a CRAM
// view slot, and died when no unpack had it mapped). A slot of the view range is ours alone.
std::atomic<std::uint64_t> zero_block_dirty{0};
void CheckZeroBlock() noexcept {
#ifdef PS5_NATIVE
    if (zero_block < 0) return;
    // A view of its own out of the range reserved for views: released only when the check has
    // finished, and no other unpack or copy can be given this address (MapCpuView's bitmap).
    const std::uintptr_t view = MapCpuView(zero_block, SparseSlot, PROT_READ);
    if (view == 0) return; // all views busy: skip the check, never touch an address we do not own
    const auto* const words = reinterpret_cast<const volatile std::uint64_t*>(view);
    for (std::size_t i = 0; i < SparseSlot / 8; i += 8)
        if (words[i] != 0) {
            if (zero_block_dirty.fetch_add(1, std::memory_order_relaxed) < 4) {
                char line[160];
                const int n = std::snprintf(line, sizeof(line), "EDEN_CRAM_ZERO_BLOCK written: word %zu is %#llx\n", i,
                                            (unsigned long long)words[i]);
                if (n > 0) Report(line);
            }
            break;
        }
    UnmapCpuView(view, SparseSlot);
#endif
}

#ifdef PS5_NATIVE
extern "C" int pthread_getname_np(pthread_t, char*, std::size_t);
#endif
void CountUnpacker(std::size_t slot) noexcept {
    char name[24] = "?";
#ifdef PS5_NATIVE
    (void)pthread_getname_np(pthread_self(), name, sizeof(name));
#endif
    name[sizeof(name) - 1] = 0;
    if ((std::strcmp(name, "GPU") == 0 || std::strcmp(name, "GPUBack") == 0) && gpu_touched != nullptr &&
        slot < gpu_cached_slots && gpu_touched[slot].exchange(1, std::memory_order_acq_rel) == 0)
        pack_gpu_marked.fetch_add(1, std::memory_order_relaxed);
    while (unpackers_lock.test_and_set(std::memory_order_acquire)) {}
    for (auto& entry : unpackers) {
        if (entry.name[0] == 0) std::memcpy(entry.name, name, sizeof(name));
        if (std::strcmp(entry.name, name) == 0) {
            entry.count.fetch_add(1, std::memory_order_relaxed);
            break;
        }
    }
    unpackers_lock.clear(std::memory_order_release);
}
void ReportUnpackers() noexcept {
    char line[512];
    int length = std::snprintf(line, sizeof(line), "EDEN_CRAM_UNPACKERS");
    for (auto& entry : unpackers) {
        if (entry.name[0] == 0 || length <= 0 || length >= static_cast<int>(sizeof(line)) - 40) break;
        length += std::snprintf(line + length, sizeof(line) - length, " %s=%u", entry.name,
                                entry.count.load(std::memory_order_relaxed));
    }
    if (length > 0 && length < static_cast<int>(sizeof(line)) - 1) {
        line[length] = '\n';
        line[length + 1] = 0;
        Report(line);
    }
}

void ColdPass(SparseRange& range, std::size_t slots) noexcept {
    CheckZeroBlock();
    const std::lock_guard pack_lock{pack_mutex};
    holding_pack = true;
    struct Held { ~Held() { holding_pack = false; } } held;
    // Room made once: a pass allocates nothing (the heap may be out of memory when it runs).
    static std::vector<std::size_t> candidates;
    if (cold_hash.size() != slots) {
        cold_hash.assign(slots, 0);
        cold_since.assign(slots, ColdNow());
        candidates.clear();
        candidates.reserve(slots);
        pack_ready.reserve(slots);
    }
    candidates.clear();
    std::uint64_t own = 0, c1 = 0, c3 = 0, c6 = 0, zero = 0, zero_old = 0;
    const std::uint32_t now = ColdNow();
    for (std::size_t slot = 0; slot < slots; ++slot) {
        if (SlotState(range, slot) != SlotOwn) {
            cold_since[slot] = now;
            cold_hash[slot] = 0;
            continue;
        }
        ++own;
        const std::uint64_t hash = ColdHash(SlotAddress(range, slot));
        if (hash != cold_hash[slot]) {
            cold_since[slot] = now;
            cold_hash[slot] = hash;
        }
        const std::uint32_t age = now - cold_since[slot];
        c1 += age >= 20;
        c3 += age >= 60;
        c6 += age >= 120;
        if (age >= ColdSeconds) candidates.push_back(slot);
        if (ColdReport() && ColdLooksZero(SlotAddress(range, slot))) {
            ++zero;
            zero_old += age >= 60;
        }
    }
    cold_zero.store(zero, std::memory_order_relaxed);
    cold_zero_old.store(zero_old, std::memory_order_relaxed);
    cold_own.store(own, std::memory_order_relaxed);
    cold_1.store(c1, std::memory_order_relaxed);
    cold_3.store(c3, std::memory_order_relaxed);
    cold_6.store(c6, std::memory_order_relaxed);
    if (PackEnabled()) PackPass(range, slots, candidates);
    if (!ColdReport()) return;
    const auto start = std::chrono::steady_clock::now();
    const bool can_compress = &::Common::Compression::CompressDataLZ4 != nullptr;
    for (std::size_t i = 0; can_compress && i < std::min(ColdSamples, candidates.size()); ++i) {
        const std::size_t slot = candidates[(cold_rotor + i * 7919) % candidates.size()];
        if (SlotState(range, slot) != SlotOwn) continue; // packed meanwhile: a read would unpack it
        std::vector<std::uint8_t> copy(SparseSlot);
        std::memcpy(copy.data(), reinterpret_cast<const void*>(SlotAddress(range, slot)), SparseSlot);
        const auto packed = ::Common::Compression::CompressDataLZ4(copy.data(), copy.size());
        cold_sampled_bytes.fetch_add(SparseSlot, std::memory_order_relaxed);
        cold_compressed_bytes.fetch_add(packed.empty() ? SparseSlot : packed.size(), std::memory_order_relaxed);
    }
    cold_rotor += ColdSamples;
    cold_compress_ns.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count()), std::memory_order_relaxed);
    const std::uint64_t sampled = cold_sampled_bytes.load(std::memory_order_relaxed);
    char line[400];
    const int length = std::snprintf(line, sizeof(line),
        "EDEN_CRAM_COLD own=%lluMiB unwritten_20s=%lluMiB unwritten_60s=%lluMiB unwritten_120s=%lluMiB "
        "lz4_sampled=%lluMiB lz4_ratio=%.3f lz4_ms=%llu zero_sampled=%lluMiB zero_unwritten_60s=%lluMiB\n",
        (unsigned long long)(own * 2), (unsigned long long)(c1 * 2), (unsigned long long)(c3 * 2),
        (unsigned long long)(c6 * 2), (unsigned long long)(sampled >> 20),
        sampled ? double(cold_compressed_bytes.load(std::memory_order_relaxed)) / double(sampled) : 0.0,
        (unsigned long long)(cold_compress_ns.load(std::memory_order_relaxed) / 1000000),
        (unsigned long long)(cold_zero.load(std::memory_order_relaxed) * 2),
        (unsigned long long)(cold_zero_old.load(std::memory_order_relaxed) * 2));
    if (length > 0) Report(line);
}

void PackPass(SparseRange& range, std::size_t slots, std::vector<std::size_t>& candidates) noexcept {
#ifdef PS5_NATIVE
    StoreReclaim();
#endif
    if (packed_slots.load(std::memory_order_acquire) == nullptr) {
        auto* table = new (std::nothrow) PackedSlot[slots]();
        if (table == nullptr) return;
        if (gpu_touched == nullptr) {
            gpu_touched = new (std::nothrow) std::atomic<std::uint8_t>[slots]();
            if (gpu_cached_slots == 0) gpu_cached_slots = slots;
        }
        if (PackKeep()) {
            pack_kept = new (std::nothrow) std::int64_t[slots];
            if (pack_kept != nullptr)
                for (std::size_t i = 0; i < slots; ++i) pack_kept[i] = -1;
        }
        packed_slots.store(table, std::memory_order_release);
        pack_times.assign(slots, 0);
        recover_order.reserve(slots);
        std::size_t scratch = static_cast<std::size_t>(LZ4_compressBound(static_cast<int>(SparseSlot)));
#ifdef EDEN_SWAP_ZSTD
        scratch = std::max(scratch, ZSTD_compressBound(SparseSlot));
#endif
        pack_scratch.resize(scratch);
    }
    if (pack_refused.load(std::memory_order_relaxed) || !PoolShortForPacking()) return;
    // Device memory nothing is bound to goes back before any guest memory is packed.
    if (&eden_radv_private_radv_ps5_eden_trim != nullptr) (void)eden_radv_private_radv_ps5_eden_trim(std::uint64_t{512} << 20);
    // Coldest first, each needing ColdSeconds << (times packed) unwritten.
    const std::uint32_t now = ColdNow();
    std::vector<std::size_t>& ready = pack_ready;
    ready.clear();
    for (const std::size_t slot : candidates)
        if (now - cold_since[slot] >= (ColdSeconds << std::min<unsigned>(pack_times[slot], 4))) ready.push_back(slot);
    std::sort(ready.begin(), ready.end(), [](std::size_t a, std::size_t b) { return cold_since[a] < cold_since[b]; });
    std::size_t done = 0;
    for (const std::size_t slot : ready) {
        if (done >= PackPerPass) break;
        if (pack_times[slot] < 255) ++pack_times[slot];
        if (PackSlot(range, slot)) {
            ++done;
            cold_since[slot] = now;
        }
    }
}

// ---- The scanner looks where something changed (2026-11-15) ----
// AllZero reads half an Own slot and ScanFine all of it (~3 MiB together): at 5 GiB held that is
// ~8 GiB of reads every pass, a second of a core and ~2 GB/s of memory bandwidth while the pool
// is short and a game loads a new area. A slot whose sampled hash did not change, and whose bytes
// the guest's kernel did not free part of, cannot have become all-zero or gained zero pages since
// the last look, which is all those two checks ask; a full content pass every FullScanPasses
// covers the rare write of zeroes that the sample misses. A skipped slot only delays a return,
// never a wrong state: ScanZero and ScanFine re-read the slot under its lock.
constexpr std::size_t FullScanPasses = 12; // a full content pass every 12 scans (24 s while the pool is short)
std::vector<std::uint64_t> scan_hash; // per slot: the sampled hash the scanner last looked at (scanner only)
// Per slot: the kernel freed part of it since the last look (set by a guest thread, cleared by the
// scanner). The table is made once, before the scanner's first pass; a guest thread that runs
// earlier finds it null and is simply caught by the hash.
std::atomic<std::atomic<std::uint8_t>*> scan_dirty{nullptr};
std::atomic<std::size_t> scan_dirty_slots{0};
std::atomic<std::uint64_t> scan_skips{0}, scan_full_passes{0};

// The scanner looks at an armed slot too, as an Own one, for the moment it takes. Without this an
// armed slot could not be given back or made Fine, and arming held on to memory the scanner would
// have freed. A touch in that moment goes uncounted; the slot keeps its age.
bool LendArmedSlot(SparseRange& range, std::size_t slot) noexcept {
#ifdef PS5_NATIVE
    LockSlot(range, slot);
    const std::uint8_t state = SlotState(range, slot);
    if (state != SlotProbe) {
        UnlockSlot(range, slot, state);
        return false;
    }
    if (mprotect(reinterpret_cast<void*>(SlotAddress(range, slot)), SparseSlot, PROT_READ | PROT_WRITE) != 0) std::abort();
    UnlockSlot(range, slot, SlotOwn);
    return true;
#else
    (void)range;
    (void)slot;
    return false;
#endif
}
void TakeArmedSlotBack(SparseRange& range, std::size_t slot, std::uint32_t armed_at) noexcept {
#ifdef PS5_NATIVE
    LockSlot(range, slot);
    const std::uint8_t state = SlotState(range, slot);
    if (state != SlotOwn) { // gone, made Fine or Packed, or armed again meanwhile
        UnlockSlot(range, slot, state);
        return;
    }
    if (mprotect(reinterpret_cast<void*>(SlotAddress(range, slot)), SparseSlot, PROT_NONE) != 0) std::abort();
    range.swap[slot] = armed_at;
    UnlockSlot(range, slot, SlotProbe);
#else
    (void)range;
    (void)slot;
    (void)armed_at;
#endif
}

void ScanPass(SparseRange& range) noexcept {
    const auto start = std::chrono::steady_clock::now();
    const std::size_t slots = (range.end - range.begin.load(std::memory_order_relaxed)) / SparseSlot;
    if (scan_hash.size() != slots) {
        scan_hash.assign(slots, 0);
        delete[] scan_dirty.load(std::memory_order_relaxed);
        auto* const marks = new (std::nothrow) std::atomic<std::uint8_t>[slots]();
        scan_dirty_slots.store(marks != nullptr ? slots : 0, std::memory_order_relaxed);
        scan_dirty.store(marks, std::memory_order_release);
        // Without the per-slot marks, look at every slot (the old way).
        if (marks == nullptr) scan_hash.clear();
    }
    auto* const dirty_marks = scan_dirty.load(std::memory_order_acquire);
    const std::uint64_t pass = counters.passes.load(std::memory_order_relaxed);
    // Duplicate sharing compares every slot's sample, so it keeps the full pass (it is off by
    // default; on the console it cost FPS and found few pairs).
    const bool full = scan_hash.empty() || dedup_requested.load(std::memory_order_relaxed) ||
                      pass % FullScanPasses == 0;
    if (full) scan_full_passes.fetch_add(1, std::memory_order_relaxed);
    std::vector<std::pair<std::uint64_t, std::size_t>> samples;
    for (std::size_t slot = 0; slot < slots; ++slot) {
        // An armed slot is lent to the scanner for this look, then armed again: reads of it would
        // fault, and a slot only ever looked at by the swapper would hold its block for good.
        std::uint32_t armed_at = 0;
        if (SlotState(range, slot) == SlotProbe) {
            if (!LendArmedSlot(range, slot)) {
                if (!scan_hash.empty()) scan_hash[slot] = 0;
                if (dirty_marks != nullptr) dirty_marks[slot].store(0, std::memory_order_relaxed);
                continue;
            }
            armed_at = range.swap[slot];
        }
        const auto rearm = [&]() noexcept {
            if (armed_at != 0) {
                TakeArmedSlotBack(range, slot, armed_at);
                armed_at = 0;
            }
        };
        if (SlotState(range, slot) != SlotOwn) {
            rearm();
            if (!scan_hash.empty()) scan_hash[slot] = 0;
            if (dirty_marks != nullptr) dirty_marks[slot].store(0, std::memory_order_relaxed);
            continue;
        }
        const std::uintptr_t at = SlotAddress(range, slot);
        const bool dirty = dirty_marks != nullptr && dirty_marks[slot].exchange(0, std::memory_order_relaxed) != 0;
        const std::uint64_t hash = ColdHash(at);
        // A pass-based Fine back-off that has come due needs its try even when nothing changed.
        const std::uint32_t fine_after = range.fine_after[slot];
        const bool fine_due = fine_after != 0 && pass >= fine_after;
        if (!full && !dirty && !fine_due && !scan_hash.empty() && hash == scan_hash[slot]) {
            scan_skips.fetch_add(1, std::memory_order_relaxed);
            rearm();
            continue;
        }
        if (!scan_hash.empty()) scan_hash[slot] = hash;
        if (AllZero(at)) { // a read without the lock: only a hint, checked again fenced
            if (ScanZero(range, slot)) counters.returns_scan.fetch_add(1, std::memory_order_relaxed);
        } else if (fine_requested.load(std::memory_order_relaxed) && ScanFine(range, slot)) {
            // now Fine
        } else if (dedup_requested.load(std::memory_order_relaxed)) {
            samples.emplace_back(SampleHash(at), slot);
        }
        rearm();
    }
    std::sort(samples.begin(), samples.end());
    for (std::size_t i = 0; i < samples.size();) {
        std::size_t j = i + 1;
        while (j < samples.size() && samples[j].first == samples[i].first) ++j;
        if (j - i > 1) {
            std::vector<std::pair<std::uint64_t, std::size_t>> full;
            for (std::size_t k = i; k < j; ++k)
                full.emplace_back(FullHash(SlotAddress(range, samples[k].second), SparseSlot), samples[k].second);
            std::sort(full.begin(), full.end());
            for (std::size_t a = 0; a < full.size();) {
                std::size_t b = a + 1;
                while (b < full.size() && full[b].first == full[a].first) {
                    (void)ScanShare(range, full[a].second, full[b].second);
                    ++b;
                }
                a = b;
            }
        }
        i = j;
    }
    if (ColdReport() || PackEnabled()) ColdPass(range, slots);
    counters.passes.fetch_add(1, std::memory_order_relaxed);
    counters.scan_ns.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count()), std::memory_order_relaxed);
}

void ServiceReport(SparseRange& range, const char* when) noexcept {
    const std::size_t slots = (range.end - range.begin.load(std::memory_order_relaxed)) / SparseSlot;
    std::size_t own = 0, shared = 0, fine = 0, fine_kept = 0;
    for (std::size_t slot = 0; slot < slots; ++slot) {
        const auto state = SlotState(range, slot);
        own += state == SlotOwn;
        shared += state == SlotShared;
        if (state == SlotFine) {
            ++fine;
            fine_kept += FineKept(range, slot);
        }
    }
    char line[640];
    const int length = std::snprintf(line, sizeof(line),
        "EDEN_CRAM %s held=%zuMiB own=%zu shared=%zu zero=%zu faults=%llu promotions=%llu copies=%llu waits=%llu "
        "returns_clear=%llu returns_free=%llu returns_scan=%llu dedup=%llu reserve_used=%llu failures=%llu "
        "passes=%llu scan_ms=%llu fine=%zu fine_held=%zuMiB fined=%llu unfined=%llu fine_failures=%llu "
        "reserve=%zuMiB host_transferred=%lluMiB freed_parts=%llu freed_parts_cleared=%lluMiB "
        "scan_skipped=%llu full_scans=%llu\n",
        when, guest_committed.load(std::memory_order_relaxed) >> 20, own, shared, slots - own - shared - fine,
        (unsigned long long)counters.faults.load(), (unsigned long long)counters.promotions.load(),
        (unsigned long long)counters.copies.load(), (unsigned long long)counters.waits.load(),
        (unsigned long long)counters.returns_clear.load(), (unsigned long long)counters.returns_free.load(),
        (unsigned long long)counters.returns_scan.load(), (unsigned long long)counters.dedup.load(),
        (unsigned long long)counters.reserve_used.load(), (unsigned long long)counters.failures.load(),
        (unsigned long long)counters.passes.load(), (unsigned long long)(counters.scan_ns.load() / 1000000),
        fine, fine_kept * FinePage >> 20, (unsigned long long)counters.fined.load(),
        (unsigned long long)counters.unfined.load(), (unsigned long long)counters.fine_failures.load(),
        reserve_count.load(std::memory_order_relaxed) * (SparseSlot >> 20),
        (unsigned long long)counters.host_reserve_used.load() * (SparseSlot >> 20),
        (unsigned long long)freed_part_calls.load(), (unsigned long long)(freed_part_bytes.load() >> 20),
        (unsigned long long)scan_skips.load(), (unsigned long long)scan_full_passes.load());
    if (length > 0) Report(line);
    // Who wrote the slots that were promoted: the largest places (offsets in the app's image).
    struct Top { const WriterSite* site; std::uint64_t count; } top[10]{};
    for (const WriterSite& site : writer_sites) {
        const std::uint64_t count = site.count.load(std::memory_order_relaxed);
        if (site.key.load(std::memory_order_relaxed) == 0 || count <= top[9].count) continue;
        std::size_t place = 9;
        while (place > 0 && top[place - 1].count < count) {
            top[place] = top[place - 1];
            --place;
        }
        top[place] = {&site, count};
    }
    char writers[640];
    int used = std::snprintf(writers, sizeof(writers),
                             "EDEN_CRAM_WRITERS promotions by code of the app %llu, by other code %llu\n",
                             (unsigned long long)promotions_in_app_code.load(), (unsigned long long)promotions_elsewhere.load());
    if (used > 0) Report(writers);
    for (const Top& entry : top) {
        if (entry.count == 0) break;
        int length = std::snprintf(writers, sizeof(writers), "EDEN_CRAM_WRITER %llu:", (unsigned long long)entry.count);
        for (const std::uintptr_t pc : entry.site->pcs) {
            if (length > 0 && static_cast<std::size_t>(length) < sizeof(writers) - 24)
                length += std::snprintf(writers + length, sizeof(writers) - length, " %llx",
                                        (unsigned long long)(pc != 0 ? pc - 0x400000ull : 0));
        }
        if (length > 0 && static_cast<std::size_t>(length) < sizeof(writers) - 2) {
            writers[length++] = '\n';
            writers[length] = 0;
            Report(writers);
        }
    }
    if (PackEnabled()) {
        char pack_line[400];
        const int pack_length = std::snprintf(pack_line, sizeof(pack_line),
            "EDEN_CRAM_PACK %s packed=%lluMiB stored=%lluMiB packs=%llu unpacks=%llu skips=%llu unpack_keep=%lluMiB"
            " kept_compared=%llu kept_differed=%llu zero_block_dirty=%llu gpu_skips=%llu gpu_marked=%llu zstd=%llu\n", when,
            (unsigned long long)(packed_count.load(std::memory_order_relaxed) * (SparseSlot >> 20)),
            (unsigned long long)(packed_bytes.load(std::memory_order_relaxed) >> 20),
            (unsigned long long)packs.load(std::memory_order_relaxed), (unsigned long long)unpacks.load(std::memory_order_relaxed),
            (unsigned long long)pack_skips.load(std::memory_order_relaxed), (unsigned long long)(unpack_keep_count.load(std::memory_order_relaxed) * 2),
            (unsigned long long)pack_keep_compared.load(std::memory_order_relaxed),
            (unsigned long long)pack_keep_differed.load(std::memory_order_relaxed),
            (unsigned long long)zero_block_dirty.load(std::memory_order_relaxed),
            (unsigned long long)pack_gpu_skips.load(std::memory_order_relaxed),
            (unsigned long long)pack_gpu_marked.load(std::memory_order_relaxed),
#ifdef EDEN_SWAP_ZSTD
            (unsigned long long)packed_zstd.load(std::memory_order_relaxed));
#else
            0ull);
#endif
        if (pack_length > 0) Report(pack_line);
        ReportUnpackers();
    }
    SwapReport(range);
}

// What finer slots would give: 16 KiB pages of the Own slots that hold only zeroes, and those that
// repeat another page (by a full hash). Read once, when the session ends.
void MeasureFinePages(SparseRange& range) noexcept {
    constexpr std::size_t Fine = 16384;
    const std::size_t slots = (range.end - range.begin.load(std::memory_order_relaxed)) / SparseSlot;
    std::vector<std::uint64_t> hashes;
    std::size_t pages = 0, zero = 0;
    for (std::size_t slot = 0; slot < slots; ++slot) {
        if (SlotState(range, slot) != SlotOwn) continue;
        for (std::size_t offset = 0; offset < SparseSlot; offset += Fine) {
            const std::uintptr_t at = SlotAddress(range, slot) + offset;
            ++pages;
            const auto* word = reinterpret_cast<const volatile std::uint64_t*>(at);
            bool is_zero = true;
            for (std::size_t i = 0; i < Fine / 8 && is_zero; ++i) is_zero = word[i] == 0;
            if (is_zero) ++zero;
            else hashes.push_back(FullHash(at, Fine));
        }
    }
    std::sort(hashes.begin(), hashes.end());
    std::size_t repeats = 0;
    for (std::size_t i = 1; i < hashes.size(); ++i) repeats += hashes[i] == hashes[i - 1];
    char line[192];
    const int length = std::snprintf(line, sizeof(line),
        "EDEN_CRAM_FINE pages16k=%zu zero=%zu (%zuMiB) repeats=%zu (%zuMiB)\n", pages, zero, zero * Fine >> 20, repeats,
        repeats * Fine >> 20);
    if (length > 0) Report(line);
}

// ---- The swapper: arms a share of the slots each tick, and sends the cold ones out ----
std::thread swapper;

// The pool's largest free block, in 2 MiB steps: what the driver and the heap can still build a
// block from. The largest free block alone stays small once the pool is in pieces, so this is read
// with 2 MiB alignment as the fork's memory watch does.
[[maybe_unused]] void SwapMeasurePool() noexcept {
#ifdef PS5_NATIVE
    const auto total = sceKernelGetDirectMemorySize();
    std::int64_t start = 0;
    std::size_t largest = 0;
    if (total > 0 && &sceKernelAvailableDirectMemorySize != nullptr &&
        sceKernelAvailableDirectMemorySize(0, total, SparseSlot, &start, &largest) == 0)
        swap_pool_largest.store(largest, std::memory_order_relaxed);
#else
    swap_pool_largest.store(0, std::memory_order_relaxed);
#endif
}
[[maybe_unused]] bool SwapPoolLow() noexcept {
    const std::size_t largest = swap_pool_largest.load(std::memory_order_relaxed);
    return largest != 0 && largest < (static_cast<std::size_t>(swap_low_bytes_mib) << 20);
}
// Memory should leave now: the pool is low, or the caller wants headroom before it is, or an
// allocation is failing right now.
[[maybe_unused]] bool SwapWanted() noexcept {
    if (swap_mode.load(std::memory_order_relaxed) < 2) return false;
    if (NowNs() < swap_urgent_until_ns.load(std::memory_order_relaxed)) return true;
    if (memory_waiters.load(std::memory_order_acquire) != 0) return true;
    const std::size_t largest = swap_pool_largest.load(std::memory_order_relaxed);
    if (largest == 0) return false;
    if (largest < (static_cast<std::size_t>(swap_low_bytes_mib) << 20)) return true;
    return swap_ahead_bytes != 0 && largest < swap_ahead_bytes;
}

// An allocation is failing (headless/heap_arenas.inc): the swapper ticks at once and sends what
// must go to the file, ignoring read-backs, for the next second.
extern "C" void eden_guest_memory_urgent(void) noexcept {
    swap_urgent_until_ns.store(NowNs() + 1'000'000'000ull, std::memory_order_relaxed);
    swap_urgent_wake.store(true, std::memory_order_relaxed);
    scanner_wake.notify_all();
}
[[maybe_unused]] void WakeSwapper() noexcept { eden_guest_memory_urgent(); }

// One round of making memory for a block outside a fault handler (a page-table slot, a thread's
// stack, the driver's ring): the same steps a fault takes, at a 10 ms pace, while the caller asks
// again for up to 15 s (src/memory_pages.cpp, src/direct_memory.c).
void MakeMemoryForBlock(unsigned round) noexcept {
    eden_guest_memory_urgent();
    RecoverForFault();
#ifdef PS5_NATIVE
    if (round % 25 == 24) (void)TrimHeapForFault(); // throttled to once per 250 ms inside
#endif
    const timespec pause{0, 10'000'000};
    nanosleep(&pause, nullptr);
}

// All the memory the swap has given back so far: progress a thread waiting for memory can see.
extern "C" std::uint64_t eden_guest_memory_given_back(void) noexcept {
    return swap_given_back.load(std::memory_order_relaxed);
}

[[maybe_unused]] bool OpenSwapFile() noexcept {
#ifdef PS5_NATIVE
    swap_free_count = 0;
    swap_next_place = 0;
    if (swap_fd < 0) swap_fd = open(swap_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    return swap_fd >= 0;
#else
    return false;
#endif
}
[[maybe_unused]] void CloseSwapFile() noexcept {
#ifdef PS5_NATIVE
    if (swap_fd < 0) return;
    (void)ftruncate(swap_fd, 0);
    (void)close(swap_fd);
    swap_fd = -1;
#endif
}

// The switches, read once a guest range starts.
void ReadSwapSwitches() noexcept {
#ifdef PS5_NATIVE
    const char* mode = std::getenv("EDEN_GUEST_SWAP");
    swap_mode.store(mode != nullptr && mode[0] == '2' ? 2 : mode != nullptr && mode[0] == '1' ? 1 : 0,
                    std::memory_order_relaxed);
    if (const char* path = std::getenv("EDEN_GUEST_SWAP_FILE")) swap_path = path;
    if (const char* mib = std::getenv("EDEN_GUEST_SWAP_MIB"))
        swap_places = static_cast<std::uint32_t>(std::min<std::size_t>(
            std::max<unsigned long>(1, std::strtoul(mib, nullptr, 10) / (SparseSlot >> 20)), SwapMostPlaces));
    if (const char* seconds = std::getenv("EDEN_GUEST_SWAP_COLD_S"))
        swap_cold_ticks = std::max<std::uint32_t>(
            static_cast<std::uint32_t>(std::strtoul(seconds, nullptr, 10) * 1000 / SwapTickMs), 2);
    if (const char* mib = std::getenv("EDEN_GUEST_SWAP_LOW_MIB"))
        swap_low_bytes_mib = static_cast<std::uint32_t>(std::strtoul(mib, nullptr, 10));
    swap_ahead_bytes = 0;
    if (const char* mib = std::getenv("EDEN_GUEST_SWAP_AHEAD_MIB"))
        swap_ahead_bytes = static_cast<std::size_t>(std::strtoul(mib, nullptr, 10)) << 20;
    const char* all = std::getenv("EDEN_GUEST_SWAP_ALL_POOLS");
    swap_all_pools = all != nullptr && all[0] == '1';
#endif
}

// A line beside EDEN_CRAM: how many slots are armed or out, where the file is and how full, how many
// went out and came back, and how long a slot took to write and to read back.
void SwapReport(SparseRange& range) noexcept {
    const int mode = swap_mode.load(std::memory_order_relaxed);
    if (mode == 0) return;
    const std::size_t slots = (range.end - range.begin.load(std::memory_order_relaxed)) / SparseSlot;
    const std::uint32_t now = swap_tick.load(std::memory_order_relaxed);
    constexpr std::uint32_t Seconds[4] = {30, 60, 120, 300};
    std::size_t probe = 0, out = 0, untouched[4] = {};
    for (std::size_t slot = 0; slot < slots; ++slot) {
        const auto state = SlotState(range, slot);
        if (state == SlotOut) {
            ++out;
            continue;
        }
        if (state != SlotProbe) continue;
        ++probe;
        for (std::size_t i = 0; i < 4; ++i) untouched[i] += (now - range.swap[slot]) * SwapTickMs / 1000 >= Seconds[i];
    }
    const auto& c = swap_counters;
    const std::uint64_t outs = c.out.load(), ins = c.in.load();
    // Places in use are the slots that are Out: the free-list counters are the swapper's, not read
    // from this thread.
    const std::size_t places_used = out;
    char line[640];
    const int length = std::snprintf(line, sizeof(line),
        "EDEN_SWAP mode=%d file=%s armed=%zuMiB out=%zuMiB in_file=%zuMiB places=%zu/%u "
        "went_out=%llu avg_out_ms=%llu came_in=%llu avg_in_ms=%llu most_in_ms=%llu failures=%llu "
        "touches=%llu returned_armed=%llu untouched_30s=%zuMiB _60s=%zuMiB _120s=%zuMiB _300s=%zuMiB "
        "pool_largest=%zuMiB ahead=%zuMiB all_pools=%d\n",
        mode, swap_path, probe * (SparseSlot >> 20), out * (SparseSlot >> 20), places_used * (SparseSlot >> 20),
        places_used, static_cast<unsigned>(swap_places), (unsigned long long)outs,
        (unsigned long long)(outs != 0 ? c.out_ns.load() / outs / 1000000 : 0), (unsigned long long)ins,
        (unsigned long long)(ins != 0 ? c.in_ns.load() / ins / 1000000 : 0),
        (unsigned long long)(c.in_most_ns.load() / 1000000), (unsigned long long)c.failures.load(),
        (unsigned long long)c.touched.load(), (unsigned long long)c.returned_armed.load(),
        untouched[0] * (SparseSlot >> 20), untouched[1] * (SparseSlot >> 20), untouched[2] * (SparseSlot >> 20),
        untouched[3] * (SparseSlot >> 20), swap_pool_largest.load(std::memory_order_relaxed) >> 20,
        swap_ahead_bytes >> 20, swap_all_pools ? 1 : 0);
    if (length > 0) Report(line);
}

// The session ends and nothing will answer a touch: armed slots are plain Own slots again. An Out
// slot stays as it is, its bytes gone with the file: the guest is down by now.
[[maybe_unused]] void DisarmSlots(SparseRange& range) noexcept {
#ifdef PS5_NATIVE
    const std::size_t slots = (range.end - range.begin.load(std::memory_order_relaxed)) / SparseSlot;
    for (std::size_t slot = 0; slot < slots; ++slot) {
        if (SlotState(range, slot) != SlotProbe) continue;
        LockSlot(range, slot);
        const std::uint8_t state = SlotState(range, slot);
        if (state == SlotProbe &&
            mprotect(reinterpret_cast<void*>(SlotAddress(range, slot)), SparseSlot, PROT_READ | PROT_WRITE) != 0)
            std::abort();
        UnlockSlot(range, slot, state == SlotProbe ? SlotOwn : state);
    }
#else
    (void)range;
#endif
}

// In the self-test, on its slot 1 (Own, `word` in it): the fault of a read and of a write on a Probe
// slot, a trip to the file and back, every byte where it was, and what the file does. A refusal
// switches the swap off and nothing else; nothing here runs with EDEN_GUEST_SWAP unset.
[[maybe_unused]] const char* SwapCheck(SparseRange& range, volatile std::uint64_t* word, char* speeds, std::size_t room) noexcept {
#ifdef PS5_NATIVE
    const std::size_t slot = 1;
    auto* whole = reinterpret_cast<volatile std::uint64_t*>(SlotAddress(range, slot));
    constexpr std::size_t Last = SparseSlot / 8 - 1;
    *word = 21;
    whole[Last] = 0x1122334455667788ull;
    if (!ArmSlot(range, slot, true) || SlotState(range, slot) != SlotProbe) return "arm";
    const std::uint64_t start = NowNs();
    const bool read_back = *word == 21;
    const std::uint64_t touch_ns = NowNs() - start;
    if (!read_back || SlotState(range, slot) != SlotOwn) return "probe-read";
    if (!ArmSlot(range, slot, true)) return "arm-again";
    *word = 22;
    if (*word != 22 || SlotState(range, slot) != SlotOwn) return "probe-write";
    // A slot given back while it is armed reads as zeroes and is written again like any other.
    if (!ArmSlot(range, slot, true)) return "arm-return";
    LockSlot(range, slot);
    ReturnSlot(range, slot);
    UnlockSlot(range, slot, SlotZero);
    if (*word != 0 || whole[Last] != 0) return "probe-return";
    *word = 22;
    whole[Last] = 0x1122334455667788ull;
    if (*word != 22 || SlotState(range, slot) != SlotOwn) return "probe-return-write";
    if (!OpenSwapFile()) {
        std::snprintf(speeds, room, "touch_ns=%llu file=none", (unsigned long long)touch_ns);
        return swap_mode.load(std::memory_order_relaxed) == 2 ? "file" : "ok";
    }
    constexpr unsigned Trips = 4;
    std::uint64_t out_ns = 0, in_ns = 0;
    for (unsigned trip = 0; trip < Trips; ++trip) {
        const auto held = guest_committed.load(std::memory_order_relaxed);
        if (!ArmSlot(range, slot, true)) return "arm-out";
        std::uint64_t mark = NowNs();
        if (!SwapOut(range, slot) || SlotState(range, slot) != SlotOut) return "out";
        out_ns += NowNs() - mark;
        if (held - guest_committed.load(std::memory_order_relaxed) != SparseSlot) return "out-held";
        mark = NowNs();
        if (trip % 2 == 0) {
            if (*word != 22 + trip) return "in-read"; // the read that brings it back
        } else {
            *word = 23 + trip; // the write that does
        }
        in_ns += NowNs() - mark;
        if (SlotState(range, slot) != SlotOwn || whole[Last] != 0x1122334455667788ull) return "in-state";
        *word = 23 + trip;
    }
    if (!ArmSlot(range, slot, true) || !SwapOut(range, slot)) return "out-last";
    LockSlot(range, slot); // a slot given back while it is out: its place too, and it reads as zeroes
    ReturnSlot(range, slot);
    UnlockSlot(range, slot, SlotZero);
    if (*word != 0 || swap_free_count != 1) return "return";
    std::snprintf(speeds, room, "touch_ns=%llu out_ms=%llu in_ms=%llu write_MBps=%llu read_MBps=%llu",
                  (unsigned long long)touch_ns, (unsigned long long)(out_ns / Trips / 1000000),
                  (unsigned long long)(in_ns / Trips / 1000000),
                  (unsigned long long)(Trips * SparseSlot * 1000ull / (out_ns + 1)),
                  (unsigned long long)(Trips * SparseSlot * 1000ull / (in_ns + 1)));
    return "ok";
#else
    (void)range, (void)word, (void)speeds, (void)room;
    return "host";
#endif
}

// One tick: the next share of the slots is armed; slots left Probe long enough are cold; while the
// pool is short the cold ones go out, the longest untouched first. `advance`: a regular tick (every
// SwapTickMs) moves the clock slot ages are counted in and arms the next share; the extra ticks
// while the pool is low only send memory out.
[[maybe_unused]] void SwapTick(SparseRange& range, std::size_t& hand,
                               std::vector<std::pair<std::int64_t, std::size_t>>& cold, bool advance) noexcept {
    const std::size_t slots = (range.end - range.begin.load(std::memory_order_relaxed)) / SparseSlot;
    if (slots == 0) return;
    SwapMeasurePool();
    const std::uint32_t now = advance ? swap_tick.fetch_add(1, std::memory_order_relaxed) + 1
                                      : swap_tick.load(std::memory_order_relaxed);
    const std::uint64_t first = swap_all_pools ? 0 : swap_pool_first.load(std::memory_order_relaxed);
    std::uint64_t end = swap_all_pools ? slots : swap_pool_end.load(std::memory_order_relaxed);
    if (end > slots) end = slots;
    const bool wanted = SwapWanted();
    if (advance && (swap_mode.load(std::memory_order_relaxed) == 1 || wanted))
        for (std::size_t i = 0; i < slots / SwapArmShare + 1; ++i, hand = (hand + 1) % slots)
            if (hand >= first && hand < end && SlotState(range, hand) == SlotOwn) (void)ArmSlot(range, hand);
    cold.clear();
    for (std::size_t slot = static_cast<std::size_t>(first); slot < end; ++slot) {
        if (SlotState(range, slot) != SlotProbe) continue;
        const std::uint32_t age = now - range.swap[slot];
        if (age >= swap_cold_ticks) cold.emplace_back(-static_cast<std::int64_t>(age), slot);
    }
    if (!wanted) return;
    // Many reads back in the last tick: what went out came straight back. One tick of rest, unless
    // an allocation is failing (then memory out beats reads coming back).
    static std::uint64_t in_before = 0;
    const std::uint64_t in_now = swap_counters.in.load(std::memory_order_relaxed);
    const bool urgent = NowNs() < swap_urgent_until_ns.load(std::memory_order_relaxed);
    const bool thrashing = in_now - in_before > SwapBackInPerTick;
    in_before = in_now;
    if (thrashing && !urgent) return;
    std::sort(cold.begin(), cold.end());
    // The lock is taken slot by slot, within a budget: a refused allocation that asks for memory
    // meanwhile gets it between two slots instead of after the whole tick.
    const std::uint64_t deadline = NowNs() + 400'000'000ull;
    const std::uint64_t failed = swap_counters.failures.load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < cold.size() && i < SwapOutPerTick && NowNs() < deadline; ++i)
        (void)SwapOut(range, cold[i].second);
    if (!urgent && swap_counters.failures.load(std::memory_order_relaxed) != failed) {
        // A drive that is full or gone: what is out stays readable, nothing more goes.
        swap_mode.store(1, std::memory_order_relaxed);
        Report("EDEN_SWAP the file refused a slot: measuring only from now on\n");
    }
}

[[maybe_unused]] void SwapLoop(SparseRange* range) {
    std::vector<std::pair<std::int64_t, std::size_t>> cold;
    cold.reserve((range->end - range->begin.load(std::memory_order_relaxed)) / SparseSlot);
    std::size_t hand = 0;
    std::uint64_t ticked_ns = NowNs();
    std::unique_lock lock{scanner_mutex};
    while (!scanner_stop) {
        // Woken at once by a failing allocation (eden_guest_memory_urgent). Every 500 ms while the
        // pool is low: one regular tick in 2 s at 400 ms sent about 6 slots a tick, and 0.8-1.1 GB
        // the game had left alone stayed in while graphics was refused memory (Ocarina, 2026-11-16).
        const bool low = SwapPoolLow();
        scanner_wake.wait_for(lock, std::chrono::milliseconds(low ? 500 : SwapTickMs), [] {
            return scanner_stop || swap_urgent_wake.exchange(false, std::memory_order_relaxed);
        });
        if (scanner_stop) break;
        lock.unlock();
        const std::uint64_t now = NowNs();
        const bool advance = now - ticked_ns >= std::uint64_t{SwapTickMs} * 1'000'000ull;
        if (advance) ticked_ns = now;
        SwapTick(*range, hand, cold, advance);
        lock.lock();
    }
}

void ScannerLoop(SparseRange* range) {
    auto next_report = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    std::unique_lock lock{scanner_mutex};
    while (!scanner_stop) {
        // Every 20 s; every 2 s while packing has a short pool to answer (the game grows ~10 MiB/s).
        // A short-pool pass now reads only the slots that changed or were freed, so the cadence can
        // be short enough to give a freed slot back within one pass instead of for a whole frame burst.
        const bool hurry = PackEnabled() && !pack_refused.load(std::memory_order_relaxed) && PoolShortForPacking();
        scanner_wake.wait_for(lock, std::chrono::seconds(hurry ? 2 : 20), [] { return scanner_stop; });
        if (scanner_stop) break;
        lock.unlock();
        if (scan_requested.load(std::memory_order_relaxed)) ScanPass(*range);
        if (std::chrono::steady_clock::now() >= next_report) {
            ServiceReport(*range, "now");
            next_report = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        }
        lock.lock();
    }
}

// Once: does a write to a slot that shows the zero block come back here, with its address, and
// resume? A refusal keeps the guest's memory committed when it is handed out, as before.
bool ServiceAvailable() noexcept {
    int state = service_state.load(std::memory_order_acquire);
    if (state != 0) return state == 1;
    const char* step = "ok";
    long long fault_ns = -1;
    void* base = nullptr;
    if (!SparseAvailable()) step = "sparse";
    else if (!InstallServiceSignal()) step = "install";
    else if ((base = NewSparseRange(2 * SparseSlot, false)) == nullptr) step = "reserve";
    if (base != nullptr) {
        SparseRange* range = nullptr;
        {
            const std::lock_guard lock{sparse_mutex};
            range = SparseRangeOf(reinterpret_cast<std::uintptr_t>(base));
        }
        if (range == nullptr || !AddStates(*range)) {
            step = "states";
        } else {
            probe_range.store(range, std::memory_order_release);
            auto* word = reinterpret_cast<volatile std::uint64_t*>(static_cast<std::uint8_t*>(base) + SparseSlot + 64);
            const auto before = counters.faults.load(std::memory_order_relaxed);
            probe_expected.store(reinterpret_cast<std::uintptr_t>(word), std::memory_order_release);
            const auto start = std::chrono::steady_clock::now();
            *word = 0x5a17ed5a17ed5a17ull; // faults once: the slot shows the zero block
            fault_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
            probe_expected.store(0, std::memory_order_release);
            if (*word != 0x5a17ed5a17ed5a17ull) step = "value";
            else if (counters.faults.load(std::memory_order_relaxed) != before + 1) step = "fault-count";
            else if (probe_mismatch.load(std::memory_order_relaxed)) step = "fault-address";
            else if (SlotState(*range, 1) != SlotOwn || SlotState(*range, 0) != SlotZero) step = "state";
            else {
                // Back to the zero block, then written again: a slot comes and goes.
                LockSlot(*range, 1);
                ReturnSlot(*range, 1);
                UnlockSlot(*range, 1, SlotZero);
                if (*word != 0) step = "return";
                else {
                    *word = 7;
                    if (*word != 7 || SlotState(*range, 1) != SlotOwn) step = "again";
                }
            }
            if (std::strcmp(step, "ok") == 0) {
                // Fenced and unfenced: read while fenced, written after.
                const std::uintptr_t at = SlotAddress(*range, 1);
                Fence(at);
                if (*word != 7) step = "fence-read";
                Unfence(at);
                *word = 9;
                if (std::strcmp(step, "ok") == 0 && *word != 9) step = "unfence";
            }
            if (std::strcmp(step, "ok") == 0) {
                // Two slots of the same bytes: one block, then copied apart by a write, then the last
                // one takes the block back.
                auto* other = reinterpret_cast<volatile std::uint64_t*>(static_cast<std::uint8_t*>(base) + 64);
                *other = 9;
                const bool shared = ScanShare(*range, 1, 0);
#ifdef PS5_NATIVE
                if (!shared || SlotState(*range, 0) != SlotShared || SlotState(*range, 1) != SlotShared) step = "share";
                else if (*other != 9 || *word != 9) step = "share-read";
                else {
                    *other = 10; // a copy of its own
                    if (*other != 10 || *word != 9 || SlotState(*range, 0) != SlotOwn) step = "share-copy";
                    else {
                        *word = 11; // the last one: the block is its own again
                        if (*word != 11 || *other != 10 || SlotState(*range, 1) != SlotOwn) step = "share-last";
                    }
                }
#else
                if (shared) step = "share-host";
#endif
            }
            if (std::strcmp(step, "ok") == 0) {
                // Slot 1 with one page of data at page 3: Fine (one 16 KiB page kept), read in place,
                // then a write makes it whole again with every byte where it was.
                LockSlot(*range, 1);
                if (SlotState(*range, 1) != SlotZero) ReturnSlot(*range, 1);
                UnlockSlot(*range, 1, SlotZero);
                auto* page3 = reinterpret_cast<volatile std::uint64_t*>(static_cast<std::uint8_t*>(base) + SparseSlot + 3 * FinePage + 8);
                *page3 = 0x0f1e2d3c4b5a6978ull;
                const auto held = guest_committed.load(std::memory_order_relaxed);
                if (!ScanFine(*range, 1) || SlotState(*range, 1) != SlotFine) step = "fine";
                else if (*page3 != 0x0f1e2d3c4b5a6978ull || *word != 0) step = "fine-read";
                else if (held - guest_committed.load(std::memory_order_relaxed) != SparseSlot - FinePage) step = "fine-held";
                else {
                    *word = 13; // page 0: the slot is whole again
                    if (*word != 13 || *page3 != 0x0f1e2d3c4b5a6978ull || SlotState(*range, 1) != SlotOwn ||
                        (range->fine[1] & 0xf00) == 0)
                        step = "fine-write";
                }
            }
#ifdef PS5_NATIVE
            if (std::strcmp(step, "ok") == 0 && swap_mode.load(std::memory_order_relaxed) != 0) {
                char speeds[160] = "";
                swap_refused = SwapCheck(*range, word, speeds, sizeof(speeds));
                swap_checked = true;
                char swap_line[256];
                const int swap_length = std::snprintf(swap_line, sizeof(swap_line), "EDEN_SWAP_CHECK ok=%d step=%s file=%s %s\n",
                                                      std::strcmp(swap_refused, "ok") == 0 ? 1 : 0, swap_refused,
                                                      swap_path, speeds);
                if (swap_length > 0) Report(swap_line);
                CloseSwapFile();
            }
#endif
            for (std::size_t slot = 0; slot < 2; ++slot)
                if (SlotState(*range, slot) != SlotZero) {
                    LockSlot(*range, slot);
                    ReturnSlot(*range, slot);
                    UnlockSlot(*range, slot, SlotZero);
                }
            probe_range.store(nullptr, std::memory_order_release);
        }
        FreeSparsePages(base, 2 * SparseSlot);
    }
    ResetCounters();
    const bool ok = std::strcmp(step, "ok") == 0;
    service_refused = step;
    char line[128];
    const int length = std::snprintf(line, sizeof(line), "EDEN_CRAM_CHECK ok=%d step=%s fault_ns=%lld\n", ok ? 1 : 0,
                                     step, fault_ns);
    if (length > 0) Report(line);
    service_state.store(ok ? 1 : 2, std::memory_order_release);
    return ok;
}

// The guest range starts being a service (when the platform passed the check).
void ServiceStart(SparseRange& range) noexcept {
    // A refusal belongs to the session it ended: a later crash must not report it again.
    refused_address.store(0, std::memory_order_relaxed);
    ReadSwapSwitches();
    if (!service_requested.load(std::memory_order_relaxed) || !ServiceAvailable() || !AddStates(range)) {
        char line[96];
        const int length = std::snprintf(line, sizeof(line), "EDEN_CRAM off: %s\n",
                                         !service_requested.load() ? "switched-off" : service_refused);
        if (length > 0) Report(line);
        return;
    }
    ResetCounters();
    range.promote = true;
#ifdef PS5_NATIVE
    FillReserve(ReserveAtStart);
#ifdef EDEN_SWAP_ZSTD
    // The contexts are made now, where allocating is fine: a fault handler only takes them.
    if (PackEnabled()) ZstdStart();
#endif
#endif
    service_range.store(&range, std::memory_order_release);
    Eden::Fastmem::guest_write_fault.store(&HandleGuestWriteFault, std::memory_order_release);
    {
        const std::lock_guard lock{scanner_mutex};
        scanner_stop = false;
    }
    scanner = std::thread(ScannerLoop, &range);
#ifdef PS5_NATIVE
    keeper = std::thread(ReserveKeeper);
    // A check that failed leaves the swap off; in measure mode a drive that cannot be written does
    // not. The core copies IPC read buffers around Probe/Out slots (hle_ipc.cpp) and says where
    // the game's pool lies (kernel.cpp), so only that pool is armed by default;
    // EDEN_GUEST_SWAP_ALL_POOLS=1 arms every Own slot instead, to measure them.
    if (swap_mode.load(std::memory_order_relaxed) != 0 && (!swap_checked || std::strcmp(swap_refused, "ok") != 0)) {
        if (!swap_checked) Report("EDEN_SWAP the self-test did not run: the swap stays off\n");
        swap_mode.store(0, std::memory_order_relaxed);
    }
    if (swap_mode.load(std::memory_order_relaxed) == 2 && !OpenSwapFile()) {
        swap_mode.store(1, std::memory_order_relaxed);
        Report("EDEN_SWAP the swap file could not be opened: measuring only from now on\n");
    }
    if (swap_mode.load(std::memory_order_relaxed) != 0) {
        swap_tick.store(SwapRearmTicks + 1, std::memory_order_relaxed);
        swap_urgent_until_ns.store(0, std::memory_order_relaxed);
        swap_urgent_wake.store(false, std::memory_order_relaxed);
        swap_given_back.store(0, std::memory_order_relaxed);
        swapped_to_file.store(0, std::memory_order_relaxed);
        for (auto& word : swap_ever_out) word.store(0, std::memory_order_relaxed);
        for (auto* counter : {&swap_counters.armed, &swap_counters.touched, &swap_counters.out, &swap_counters.in,
                              &swap_counters.out_ns, &swap_counters.in_ns, &swap_counters.in_most_ns,
                              &swap_counters.failures, &swap_counters.returned_armed})
            counter->store(0, std::memory_order_relaxed);
        for (auto& age : swap_counters.age) age.store(0, std::memory_order_relaxed);
        try {
            swapper = std::thread(SwapLoop, &range);
        } catch (...) {
            swap_mode.store(1, std::memory_order_relaxed);
            Report("EDEN_SWAP no thread for the swapper: measuring only from now on\n");
        }
        char swap_line[256];
        const int swap_length = std::snprintf(swap_line, sizeof(swap_line),
            "EDEN_SWAP on: mode=%d file=%s places=%u cold=%us low=%uMiB ahead=%zuMiB all_pools=%d\n",
            swap_mode.load(std::memory_order_relaxed), swap_path, static_cast<unsigned>(swap_places),
            static_cast<unsigned>(swap_cold_ticks * SwapTickMs / 1000), static_cast<unsigned>(swap_low_bytes_mib),
            swap_ahead_bytes >> 20, swap_all_pools ? 1 : 0);
        if (swap_length > 0) Report(swap_line);
    }
#endif
    char line[128];
    const int length = std::snprintf(line, sizeof(line), "EDEN_CRAM on: scan=%d dedup=%d fine=%d reserve=%zuMiB\n",
                                     scan_requested.load() ? 1 : 0, dedup_requested.load() ? 1 : 0,
                                     fine_requested.load() ? 1 : 0,
                                     reserve_count.load() * (SparseSlot >> 20));
    if (length > 0) Report(line);
}

void ServiceStop(SparseRange& range) noexcept {
    if (service_range.load(std::memory_order_acquire) != &range) return;
    {
        const std::lock_guard lock{scanner_mutex};
        scanner_stop = true;
    }
    scanner_wake.notify_all();
    if (scanner.joinable()) scanner.join();
#ifdef PS5_NATIVE
    if (swapper.joinable()) swapper.join();
    if (keeper.joinable()) keeper.join();
#endif
    ServiceReport(range, "end");
    MeasureFinePages(range);
#ifdef PS5_NATIVE
    DisarmSlots(range);
#endif
    service_range.store(nullptr, std::memory_order_release);
    // Packed data goes with the range (ReleaseServiceBlocks drops what is still packed after this).
#ifdef PS5_NATIVE
    EmptyReserve();
    CloseSwapFile();
#endif
}

// What a guest range keeps when it is given back: each block once, shared ones too.
void ReleaseServiceBlocks(SparseRange& range) noexcept {
#ifdef PS5_NATIVE
    const std::size_t slots = (range.end - range.begin.load(std::memory_order_relaxed)) / SparseSlot;
    for (std::size_t slot = 0; slot < slots; ++slot) {
        DropSlotAliases(range, slot);
        const auto state = SlotState(range, slot);
        if (state == SlotFine) {
            const std::size_t kept = FineKept(range, slot);
            if (kept != 0 && sceKernelReleaseDirectMemory(range.physical[slot], kept * FinePage) != 0) std::abort();
            guest_committed.fetch_sub(kept * FinePage, std::memory_order_relaxed);
        } else if (state == SlotOwn || state == SlotProbe) {
            if (sceKernelReleaseDirectMemory(range.physical[slot], SparseSlot) != 0) std::abort();
        } else if (state == SlotPacked) {
            ForgetPacked(slot);
        } else if (state == SlotOut) {
            // No block: its place in the file goes with the session (the file is truncated at stop).
        } else if (state == SlotShared && UnshareBlock(range.physical[slot]) == 0) {
            if (sceKernelReleaseDirectMemory(range.physical[slot], SparseSlot) != 0) std::abort();
            guest_committed.fetch_sub(SparseSlot, std::memory_order_relaxed);
        }
        if (state == SlotOwn || state == SlotProbe) guest_committed.fetch_sub(SparseSlot, std::memory_order_relaxed);
        range.owned[slot].store(false, std::memory_order_relaxed);
    }
    if (&range == guest_range.load(std::memory_order_acquire) || service_range.load(std::memory_order_acquire) == nullptr) {
        store_current = StoreBlocks; // every packed slot was forgotten above: all blocks go back
        StoreReclaim();
        delete[] packed_slots.exchange(nullptr, std::memory_order_acq_rel);
        cold_hash.clear();
        cold_since.clear();
        pack_times.clear();
    }
#else
    const std::size_t slots = (range.end - range.begin.load(std::memory_order_relaxed)) / SparseSlot;
    for (std::size_t slot = 0; slot < slots; ++slot) {
        if (SlotState(range, slot) == SlotOwn) guest_committed.fetch_sub(SparseSlot, std::memory_order_relaxed);
        if (SlotState(range, slot) == SlotFine) guest_committed.fetch_sub(FineKept(range, slot) * FinePage, std::memory_order_relaxed);
        range.owned[slot].store(false, std::memory_order_relaxed);
    }
#endif
}
} // namespace

// Whether a slot may be without a mapping (Probe, Out): a system call must not be given a pointer
// into the guest's memory then. The core copies IPC read buffers while this is true (hle_ipc.cpp,
// CopiedIfSwapping).
extern "C" bool eden_guest_swap_active(void) noexcept {
    return swap_mode.load(std::memory_order_relaxed) != 0;
}

// Where the game's pool lies in the guest's backing, when the guest's kernel says so (kernel.cpp,
// InitializeMemoryLayout). Whole slots only: a slot it shares with the applets' pool stays unarmed
// unless EDEN_GUEST_SWAP_ALL_POOLS=1.
extern "C" void eden_guest_swap_pool(std::uint64_t offset, std::uint64_t bytes) noexcept {
    swap_pool_first.store((offset + SparseSlot - 1) / SparseSlot, std::memory_order_relaxed);
    swap_pool_end.store((offset + bytes) / SparseSlot, std::memory_order_relaxed);
}

// The slot behind a host address of the guest's memory: its state, and in *ever_out whether it ever
// went out. -1 for an address outside the guest's range or with no service running.
extern "C" int eden_guest_swap_address_state(const void* host, int* ever_out) noexcept {
    if (ever_out != nullptr) *ever_out = 0;
    SparseRange* const range = guest_range.load(std::memory_order_acquire);
    if (range == nullptr || range->state == nullptr) return -1;
    const auto at = reinterpret_cast<std::uintptr_t>(host);
    const std::uintptr_t begin = range->begin.load(std::memory_order_relaxed);
    if (at < begin || at >= range->end) return -1;
    const std::size_t slot = static_cast<std::size_t>((at - begin) / SparseSlot);
    if (ever_out != nullptr && slot < SwapEverOutSlots)
        *ever_out = static_cast<int>((swap_ever_out[slot / 64].load(std::memory_order_relaxed) >> (slot % 64)) & 1);
    return SlotState(*range, slot);
}

// The driver ran out of device memory (vulkan_memory_allocator.cpp): packs the coldest slots now,
// those unwritten for 10 s or more, until `bytes` went back. Returns what went back.
std::atomic<std::uint64_t> recover_calls{0}, recovered_bytes{0};
namespace {
#ifdef PS5_NATIVE
// The last resort of RecoverMemory: blocks the service holds idle (kept for unpacking, then the
// guest's reserve) go back to the pool. Late in a long session without Pro memory the heap and
// the JIT were refused their next piece and the app ended, while 768 MiB sat in these two; a
// guest write or unpack that later finds the pool empty asks for packing as before.
std::size_t GiveBackIdleBlocks(std::size_t bytes) noexcept {
    // These are the guest's emergency blocks, not spare memory: a writer waiting for a block
    // (memory_waiters) takes one from the reserve or the unpack keep, and a pack loop in flight
    // may need them next. On 2026-11-14 the heap's recovery gave back 96 MiB here while a write
    // fault was waiting for one block; that slot waited 2 s and the app died. Keep both while
    // anyone is waiting, and keep half of each otherwise (the rule TakeHostReserveBlock uses).
    if (memory_waiters.load(std::memory_order_acquire) != 0) return 0;
    if (recover_active.load(std::memory_order_acquire) != 0 && !holding_pack) return 0;
    const std::size_t keep_unpack = UnpackKeepTarget() / 2;
    const std::size_t keep_reserve = ReserveTarget() / 2;
    std::size_t freed = 0;
    while (freed < bytes) {
        std::int64_t physical = -1;
        Spin(unpack_keep_lock);
        if (const std::size_t count = unpack_keep_count.load(std::memory_order_relaxed);
            count > keep_unpack) {
            physical = unpack_keep[count - 1];
            unpack_keep_count.store(count - 1, std::memory_order_relaxed);
        }
        unpack_keep_lock.clear(std::memory_order_release);
        if (physical < 0) {
            Spin(reserve_lock);
            if (const std::size_t count = reserve_count.load(std::memory_order_relaxed);
                count > keep_reserve) {
                physical = reserve[count - 1];
                reserve_count.store(count - 1, std::memory_order_relaxed);
            }
            reserve_lock.clear(std::memory_order_release);
        }
        if (physical < 0) break;
        if (sceKernelReleaseDirectMemory(physical, SparseSlot) != 0) std::abort();
        freed += SparseSlot;
    }
    if (freed != 0) {
        idle_given_back.fetch_add(freed, std::memory_order_relaxed);
        if (idle_give_back_calls.fetch_add(1, std::memory_order_relaxed) < 24) {
            char line[96];
            const int length = std::snprintf(line, sizeof(line), "EDEN_CRAM idle blocks given back: %zu MiB\n", freed >> 20);
            if (length > 0) Report(line);
        }
    }
    return freed;
}
#endif
std::size_t RecoverMemory(std::size_t bytes, bool wait) noexcept {
#ifdef PS5_NATIVE
    // An allocation is failing: the swapper ticks at once and sends what must go while this packing
    // runs, and for a second after (the fork's eden_guest_memory_urgent).
    WakeSwapper();
    SparseRange* const range = service_range.load(std::memory_order_acquire);
    // Device memory no resource uses goes back first: it costs nothing later (packing makes the
    // guest fault its data back).
    std::size_t trimmed = 0;
    if (&eden_radv_private_radv_ps5_eden_trim != nullptr)
        trimmed = static_cast<std::size_t>(eden_radv_private_radv_ps5_eden_trim(bytes));
    if (trimmed >= bytes) return trimmed;
    bytes -= trimmed;
    if (range == nullptr || !PackEnabled() || pack_refused.load(std::memory_order_relaxed) || holding_pack)
        return trimmed + GiveBackIdleBlocks(bytes);
    std::unique_lock pack_lock{pack_mutex, std::defer_lock};
    if (wait) pack_lock.lock();
    else if (!pack_lock.try_lock()) return trimmed + GiveBackIdleBlocks(bytes);
    holding_pack = true;
    struct Held {
        Held() { recover_active.fetch_add(1, std::memory_order_relaxed); }
        ~Held() {
            recover_active.fetch_sub(1, std::memory_order_relaxed);
            holding_pack = false;
        }
    } held;
    if (packed_slots.load(std::memory_order_acquire) == nullptr || cold_since.empty() ||
        recover_order.capacity() < cold_since.size())
        return trimmed + GiveBackIdleBlocks(bytes);
    const std::uint32_t now = ColdNow();
    recover_order.clear();
    for (std::size_t slot = 0; slot < cold_since.size(); ++slot)
        // Cold for 10 s, twice as long for each time the slot was packed or would not pack (the
        // scanner's back-off): a slot that does not compress was compressed again on every call,
        // and Ocarina of Time's memory mostly does not (LZ4 ratio ~0.83).
        if (SlotState(*range, slot) == SlotOwn &&
            now - cold_since[slot] >= (10u << std::min<unsigned>(pack_times[slot], 4)))
            recover_order.push_back(slot);
    std::sort(recover_order.begin(), recover_order.end(),
              [](std::size_t a, std::size_t b) { return cold_since[a] < cold_since[b]; });
    std::size_t freed = 0;
    for (const std::size_t slot : recover_order) {
        if (freed >= bytes) break;
        if (PackSlot(*range, slot)) {
            freed += SparseSlot;
            cold_since[slot] = now;
            if (pack_times[slot] < 255) ++pack_times[slot];
        } else {
            // Would not pack (or could not now): not tried again until it has waited its back-off.
            cold_since[slot] = now;
            if (pack_times[slot] < 255) ++pack_times[slot];
        }
    }
    const auto calls = recover_calls.fetch_add(1, std::memory_order_relaxed);
    recovered_bytes.fetch_add(freed, std::memory_order_relaxed);
    if (calls < 24) {
        char line[160];
        const int length = std::snprintf(line, sizeof(line), "EDEN_CRAM_PACK recover%s asked=%zuMiB packed=%zuMiB\n",
                                          wait ? "" : "_try", bytes >> 20, freed >> 20);
        if (length > 0) Report(line);
    }
    if (freed < bytes) freed += GiveBackIdleBlocks(bytes - freed);
    return trimmed + freed;
#else
    (void)bytes;
    (void)wait;
    return 0;
#endif
}
} // namespace

// Waits for the scanner if it is packing (the driver's allocations).
extern "C" std::size_t eden_memory_recover(std::size_t bytes) noexcept { return RecoverMemory(bytes, true); }
// Never waits: for the heap, which may hold its growth lock while the scanner allocates.
extern "C" std::size_t eden_memory_recover_try(std::size_t bytes) noexcept { return RecoverMemory(bytes, false); }


// ---- What the guest's kernel tells the service ----

// Development switches (dev-settings cram=off, cram_scan=off, cram_dedup=off), before the guest's
// memory is taken.
void SetGuestMemoryService(bool on, bool scan, bool dedup) noexcept {
    service_requested.store(on, std::memory_order_relaxed);
    scan_requested.store(scan, std::memory_order_relaxed);
    dedup_requested.store(dedup, std::memory_order_relaxed);
}

// Settings > Performance > Compact memory (off unless chosen): written slots with many pages of
// zeroes keep only the others. It gives back more memory, and a game that writes such a slot again
// waits for it to be made whole (a fault and a 2 MiB copy, about 0.3 ms).
void SetGuestMemoryFine(bool on) noexcept {
    fine_requested.store(on, std::memory_order_relaxed);
}

// The guest's kernel hands pages out: they are committed now only without the service.
void CommitGuestMemoryOnHandOut(std::uint64_t offset, std::size_t bytes) noexcept {
    SparseRange* const range = guest_range.load(std::memory_order_acquire);
    if (range != nullptr && range->promote) return;
    CommitGuestMemory(offset, bytes);
}

// The guest's kernel fills pages (`at` is the backing at `offset`). With the service, a fill with
// zeroes leaves Zero slots alone and gives whole Own or Shared slots back; anything else is written.
void ClearGuestMemory(std::uint8_t* at, std::uint64_t offset, std::size_t bytes, int fill) noexcept {
    SparseRange* const range = guest_range.load(std::memory_order_acquire);
    if (range == nullptr || !range->promote || fill != 0) {
        if (range != nullptr && !range->promote) CommitGuestMemory(offset, bytes);
        std::memset(at, fill, bytes); // (with the service, a write to a Zero slot promotes it)
        return;
    }
    const std::uint64_t end = offset + bytes;
    for (std::uint64_t cursor = offset; cursor < end;) {
        const std::size_t slot = static_cast<std::size_t>(cursor / SparseSlot);
        const std::uint64_t slot_end = (cursor / SparseSlot + 1) * SparseSlot;
        const std::uint64_t piece_end = std::min(end, slot_end);
        const bool whole = cursor % SparseSlot == 0 && piece_end == slot_end;
        if (whole) {
            LockSlot(*range, slot);
            const auto state = SlotState(*range, slot);
            if (state != SlotZero) {
                ReturnSlot(*range, slot);
                counters.returns_clear.fetch_add(1, std::memory_order_relaxed);
            }
            UnlockSlot(*range, slot, SlotZero);
        } else if (SlotState(*range, slot) != SlotZero) {
            std::memset(at + (cursor - offset), 0, static_cast<std::size_t>(piece_end - cursor));
        }
        cursor = piece_end;
    }
}

// A page the guest's kernel fills one at a time (optimized allocation): the same rule.
void FillGuestMemory(std::uint8_t* at, int fill, std::size_t bytes) noexcept {
    SparseRange* const range = guest_range.load(std::memory_order_acquire);
    if (range != nullptr && range->promote && fill == 0) {
        const auto begin = range->begin.load(std::memory_order_relaxed);
        const auto address = reinterpret_cast<std::uintptr_t>(at);
        if (address >= begin && address < range->end && SlotState(*range, (address - begin) / SparseSlot) == SlotZero)
            return;
    }
    std::memset(at, fill, bytes);
}

// The guest's kernel freed pages (a free block of 2 MiB or more in its page heap): every slot they
// cover whole goes back. EDEN_GUEST_FREE_RETURN=0 keeps freed slots held until they are reused.
bool ReturnFreedSlots() noexcept {
    static const bool on = [] {
        const char* value = std::getenv("EDEN_GUEST_FREE_RETURN");
        return value == nullptr || value[0] != '0';
    }();
    return on;
}

void GuestMemoryFreed(std::uint64_t offset, std::size_t bytes) noexcept {
    SparseRange* const range = guest_range.load(std::memory_order_acquire);
    if (range == nullptr || !range->promote || bytes < SparseSlot || !ReturnFreedSlots()) return;
    const std::uint64_t first = (offset + SparseSlot - 1) / SparseSlot;
    const std::uint64_t last = (offset + bytes) / SparseSlot; // exclusive
    for (std::uint64_t slot = first; slot < last; ++slot) {
        LockSlot(*range, static_cast<std::size_t>(slot));
        if (SlotState(*range, static_cast<std::size_t>(slot)) != SlotZero) {
            ReturnSlot(*range, static_cast<std::size_t>(slot));
            counters.returns_free.fetch_add(1, std::memory_order_relaxed);
        }
        UnlockSlot(*range, static_cast<std::size_t>(slot), SlotZero);
    }
}

// The guest's kernel freed pages that do not make a free block of 2 MiB (a region's tail, a page
// group's blocks): GuestMemoryFreed never sees them, and the slot they sit in stays held whole.
// MKW unmaps ~1.7 GB in a 5-minute session in pieces of ~370 KiB, and ~640 MiB of it was never
// given back. Nothing in the guest can read freed pages (the kernel fills or hands them out again
// before use), so they are cleared where the slot has a block of its own; the scanner then gives
// back the slot (all zero) or its pages of zeroes (Fine). EDEN_GUEST_FREE_ZERO=0 leaves them.
bool ZeroFreedParts() noexcept {
    static const bool on = [] {
        const char* value = std::getenv("EDEN_GUEST_FREE_ZERO");
        return value == nullptr || value[0] != '0';
    }();
    return on;
}

void GuestMemoryFreedPart(std::uint64_t offset, std::size_t bytes) noexcept {
    SparseRange* const range = guest_range.load(std::memory_order_acquire);
    if (range == nullptr || !range->promote || bytes == 0 || !ZeroFreedParts()) return;
    const std::size_t slots = (range->end - range->begin.load(std::memory_order_relaxed)) / SparseSlot;
    freed_part_calls.fetch_add(1, std::memory_order_relaxed);
    for (std::uint64_t at = offset, end = offset + bytes; at < end;) {
        const std::size_t slot = static_cast<std::size_t>(at / SparseSlot);
        const std::uint64_t slot_end = std::min<std::uint64_t>(end, (at / SparseSlot + 1) * SparseSlot);
        if (slot >= slots) break;
        LockSlot(*range, slot);
        const std::uint8_t state = SlotState(*range, slot);
        if (state == SlotOwn) {
            std::memset(reinterpret_cast<void*>(SlotAddress(*range, slot) + at % SparseSlot), 0,
                        static_cast<std::size_t>(slot_end - at));
            freed_part_bytes.fetch_add(slot_end - at, std::memory_order_relaxed);
            // The scanner looks at this slot next pass even if the sampled hash did not move: an
            // 11 KiB free (the usual size) usually misses every sampled word.
            if (auto* const marks = scan_dirty.load(std::memory_order_acquire);
                marks != nullptr && slot < scan_dirty_slots.load(std::memory_order_relaxed))
                marks[slot].store(1, std::memory_order_relaxed);
        }
        UnlockSlot(*range, slot, state);
        at = slot_end;
    }
}

// The graphics caches hold the real contents of [host, host + bytes): images only the GPU wrote,
// whose guest pages are a second copy nothing reads before the caches write it back (a CPU read
// downloads them first; a CPU write gets a fresh zeroed block). Every slot the range covers whole
// goes back to the zero block, as a slot the guest's kernel clears; by_state[s] counts the slots
// found in state s (Zero, Own, Shared, Fine, Packed, Probe, Out). Bytes given back.
extern "C" std::size_t eden_guest_memory_discard(const void* host, std::size_t bytes,
                                                 std::uint64_t* by_state) noexcept {
    SparseRange* const range = guest_range.load(std::memory_order_acquire);
    if (range == nullptr || range->state == nullptr || !range->promote || bytes < SparseSlot) return 0;
    const auto at = reinterpret_cast<std::uintptr_t>(host);
    const std::uintptr_t begin = range->begin.load(std::memory_order_relaxed);
    if (at < begin || at >= range->end) return 0;
    const std::uint64_t offset = at - begin;
    const std::uint64_t end = std::min<std::uint64_t>(offset + bytes, range->end - begin);
    const std::uint64_t first = (offset + SparseSlot - 1) / SparseSlot;
    const std::uint64_t last = end / SparseSlot; // exclusive
    const std::size_t before = guest_committed.load(std::memory_order_relaxed);
    for (std::uint64_t slot = first; slot < last; ++slot) {
        const auto index = static_cast<std::size_t>(slot);
        // Never wait for a slot: a write fault holds one's lock while it waits for a block (up to
        // 2 s, WaitForBlock), and the caller is the thread that draws. A slot left locked is kept.
        if (!TryLockSlot(*range, index)) continue;
        const std::uint8_t state = SlotState(*range, index);
        if (by_state != nullptr && state <= SlotOut) ++by_state[state];
        if (state != SlotZero) {
            ReturnSlot(*range, index);
            counters.returns_clear.fetch_add(1, std::memory_order_relaxed);
        }
        UnlockSlot(*range, index, SlotZero);
    }
    const std::size_t after = guest_committed.load(std::memory_order_relaxed);
    return before > after ? before - after : 0;
}

// Host memory behind [offset, offset + bytes) of the guest's memory (the memory report splits it
// by the guest kernel's pools). Whole slots count 2 MiB, Fine slots their kept pages.
std::size_t GuestMemoryHeldIn(std::uint64_t offset, std::size_t bytes) noexcept {
    SparseRange* const range = guest_range.load(std::memory_order_acquire);
    if (range == nullptr || range->state == nullptr || bytes == 0) return 0;
    const std::size_t slots = (range->end - range->begin.load(std::memory_order_relaxed)) / SparseSlot;
    std::size_t held = 0;
    const std::size_t last = std::min<std::size_t>(slots, (offset + bytes + SparseSlot - 1) / SparseSlot);
    for (std::size_t slot = static_cast<std::size_t>(offset / SparseSlot); slot < last; ++slot) {
        const auto state = SlotState(*range, slot);
        // Probe keeps its Own block; Out and Packed hold nothing; Zero nothing.
        if (state == SlotOwn || state == SlotShared || state == SlotProbe) held += SparseSlot;
        else if (state == SlotFine) held += FineKept(*range, slot) * FinePage;
    }
    return held;
}

// For the host checks: one scan pass now, and a slot's state (0 Zero, 1 Own, 2 Shared, -1 none).
void GuestMemoryScanNow() noexcept {
    if (SparseRange* const range = service_range.load(std::memory_order_acquire)) ScanPass(*range);
}
int GuestMemorySlotState(std::uint64_t offset) noexcept {
    SparseRange* const range = guest_range.load(std::memory_order_acquire);
    if (range == nullptr || range->state == nullptr) return -1;
    return SlotState(*range, static_cast<std::size_t>(offset / SparseSlot));
}

// ---- The 64-bit fastmem window (src/host_memory.cpp, Window64; backported from ProsperoEden) ----

// From a JIT fault in the window: shows the slot at `backing` (2 MiB aligned) at `window_at`. An Own
// slot shows its block, read-write when `writable`; a Zero or Shared slot shows its block read-only
// for a read. A write to a slot without a block of its own sets `promote` and shows nothing: the
// caller promotes it as a write to the backing would and asks again. False otherwise (the access
// then takes the page-table path).
bool AliasGuestSlot(std::uintptr_t backing, std::uintptr_t window_at, bool writable, bool write, bool* promote) noexcept {
#ifdef PS5_NATIVE
    auto* range = guest_range.load(std::memory_order_acquire);
    if (!range || !range->state || !range->aliases) return false;
    const std::uintptr_t begin = range->begin.load(std::memory_order_relaxed);
    if (backing < begin || backing >= range->end || (backing - begin) % SparseSlot || window_at % SparseSlot) return false;
    const std::size_t slot = (backing - begin) / SparseSlot;
    LockSlot(*range, slot);
    const std::uint8_t state = SlotState(*range, slot);
    std::int64_t physical = -1;
    if (state == SlotOwn) {
        physical = range->physical[slot];
    } else if (write) {
        *promote = true;
    } else if (state == SlotZero) {
        physical = zero_block, writable = false;
    } else if (state == SlotShared) {
        physical = range->physical[slot], writable = false;
    }
    bool shown = false;
    if (physical >= 0) {
        const int protection = writable ? PROT_READ | PROT_WRITE : PROT_READ;
        const std::uintptr_t entry = window_at | (writable ? 0 : AliasReadOnly);
        std::uintptr_t* free_entry = nullptr;
        bool known = false;
        for (unsigned i = 0; i < AliasesPerSlot; ++i) {
            auto& alias = range->aliases[slot * AliasesPerSlot + i];
            if ((alias & ~AliasFlags) == window_at) {
                // Another core showed it first, or it shows it read-only or in pieces.
                known = true;
                shown = alias == entry || (MapSlot(window_at, protection, physical) && ((alias = entry), true));
                break;
            }
            if (!alias && !free_entry) free_entry = &alias;
        }
        if (!known && free_entry && MapSlot(window_at, protection, physical)) {
            *free_entry = entry;
            shown = true;
            aliases_made.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (!shown) aliases_refused.fetch_add(1, std::memory_order_relaxed);
    UnlockSlot(*range, slot, state);
    return shown;
#else
    (void)backing, (void)window_at, (void)writable, (void)write, (void)promote;
    return false; // the host's guest memory is private: nothing to alias
#endif
}

// From a JIT fault in a window region that cannot show a whole block (not on a 2 MiB boundary of the
// backing, or partly protected): shows the 16 KiB at `backing` of an Own slot at `at`, inside the
// window region `region_at`. The slot's entry names the whole region, so any change of the slot voids
// it. A write to a slot that is not Own sets `promote`, as for AliasGuestSlot.
bool AliasGuestPage(std::uintptr_t backing, std::uintptr_t at, std::uintptr_t region_at, bool writable, bool write,
                    bool* promote) noexcept {
#ifdef PS5_NATIVE
    auto* range = guest_range.load(std::memory_order_acquire);
    if (!range || !range->state || !range->aliases) return false;
    const std::uintptr_t begin = range->begin.load(std::memory_order_relaxed);
    if (backing < begin || backing >= range->end || (backing - begin) % FinePage || at % FinePage) return false;
    const std::size_t slot = (backing - begin) / SparseSlot;
    LockSlot(*range, slot);
    const std::uint8_t state = SlotState(*range, slot);
    bool shown = false;
    // A Zero slot shows the zero block and a Shared one its shared block, read-only; a write there
    // promotes the slot first (the caller). A Fine slot keeps its pages apart: not shown.
    const std::size_t offset = backing - SlotAddress(*range, slot);
    std::int64_t physical = -1;
    if (state == SlotOwn || (!write && state == SlotShared)) {
        physical = range->physical[slot] >= 0 ? range->physical[slot] + std::int64_t(offset) : -1;
    } else if (!write && state == SlotZero) {
        physical = zero_block + std::int64_t(offset);
    } else if (write) {
        *promote = true;
    }
    if (state != SlotOwn) writable = false;
    if (physical >= 0) {
        std::uintptr_t* entry = nullptr;
        bool known = false;
        for (unsigned i = 0; i < AliasesPerSlot && !known; ++i) {
            auto& alias = range->aliases[slot * AliasesPerSlot + i];
            if ((alias & ~AliasFlags) == region_at) entry = &alias, known = true;
            else if (!alias && !entry) entry = &alias;
        }
        void* address = reinterpret_cast<void*>(at);
        if (entry && sceKernelMapDirectMemory(&address, FinePage, writable ? PROT_READ | PROT_WRITE : PROT_READ, MAP_FIXED,
                                              physical, FinePage) == 0 && address == reinterpret_cast<void*>(at)) {
            if (!known) aliases_made.fetch_add(1, std::memory_order_relaxed);
            *entry = region_at | AliasSmall;
            shown = true;
        }
    }
    if (!shown) aliases_refused.fetch_add(1, std::memory_order_relaxed);
    UnlockSlot(*range, slot, state);
    return shown;
#else
    (void)backing, (void)at, (void)region_at, (void)writable, (void)write, (void)promote;
    return false;
#endif
}

// The window region at `window_at` no longer shows the slot at `backing` (guest unmap, protection).
// True when it did.
bool DropGuestAlias(std::uintptr_t backing, std::uintptr_t window_at) noexcept {
    auto* range = guest_range.load(std::memory_order_acquire);
    if (!range || !range->state || !range->aliases) return false;
    const std::uintptr_t begin = range->begin.load(std::memory_order_relaxed);
    if (backing < begin || backing >= range->end) return false;
    const std::size_t slot = (backing - begin) / SparseSlot;
    bool dropped = false;
    LockSlot(*range, slot);
    for (unsigned i = 0; i < AliasesPerSlot; ++i)
        if (auto& alias = range->aliases[slot * AliasesPerSlot + i]; alias && (alias & ~AliasFlags) == window_at) {
            VoidWindowRegion(window_at);
            alias = 0;
            dropped = true;
            aliases_dropped.fetch_add(1, std::memory_order_relaxed);
        }
    UnlockSlot(*range, slot, SlotState(*range, slot));
    return dropped;
}

void GuestAliasCounts(std::uint64_t* made, std::uint64_t* dropped, std::uint64_t* refused) noexcept {
    *made = aliases_made.load(std::memory_order_relaxed);
    *dropped = aliases_dropped.load(std::memory_order_relaxed);
    *refused = aliases_refused.load(std::memory_order_relaxed);
}
