// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <span>
#include <type_traits>

#include "common/cityhash.h"
#include "common/common_types.h"

namespace VideoCommon {

/// What a buffer cache runtime offers for stream uniform reuse.
///  - StreamUniformRef: a small trivially copyable description of one stream allocation (the
///    host buffer, its offset and whatever else the runtime needs to add it to a descriptor
///    payload again, e.g. the device address of the descriptor buffer path).
///  - LastStreamUniformRef(): the allocation made by the latest BindMappedUniformBuffer call.
///  - RebindStreamUniform(ref, size): adds `ref` to the descriptor payload exactly as
///    BindMappedUniformBuffer added the allocation the first time.
template <class R>
concept StreamUniformReusable = requires(R& runtime, u32 size) {
    typename R::StreamUniformRef;
    requires std::is_trivially_copyable_v<typename R::StreamUniformRef>;
    { runtime.LastStreamUniformRef() } -> std::same_as<typename R::StreamUniformRef>;
    runtime.RebindStreamUniform(runtime.LastStreamUniformRef(), size);
    { runtime.CurrentTick() } -> std::convertible_to<u64>;
};

/// EDEN_NX2_UBOREUSE=0 turns the reuse off (every small uniform buffer is copied, as before).
[[nodiscard]] inline bool UniformStreamReuseEnabled() noexcept {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_UBOREUSE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

/// EDEN_NX2_UBO_HISTORY=0 keeps one previous bind per slot (the pre-#2273 behaviour), for A/B
/// runs. With it on, a slot keeps a 16-way history of recent binds, so actors that alternate
/// their constant buffers every draw hit instead of copying again.
[[nodiscard]] inline bool UniformStreamHistoryEnabled() noexcept {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_UBO_HISTORY");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

template <class Ref>
class UniformStreamReuse {
public:
    static constexpr u32 MaxBytes = 4096; // BufferCache's DEFAULT_SKIP_CACHE_SIZE
    static constexpr size_t NumStages = 5;
    static constexpr size_t NumSlots = 18; // Maxwell constant buffers per stage
    /// A slot's 16-entry history bounds the actors that can alternate on one binding.
    static constexpr size_t HistoryWays = 16;
    /// With the history on, a larger miss tolerance stops a normal pack cycle from dropping a
    /// slot into the copy-without-compare path, and a shorter DirectBinds streak returns it to
    /// comparing sooner.
    static constexpr u16 MissStreak = 32;
    static constexpr u16 DirectBinds = 16;
    /// The values the single-slot path used before the history existed (EDEN_NX2_UBO_HISTORY=0).
    static constexpr u16 MissStreakSingle = 8;
    static constexpr u16 DirectBindsSingle = 64;

    struct HistoryEntry {
        Ref ref{};
        u64 tick{};
        u64 device_addr{};
        u32 size{};
        u64 hash{};
        std::array<u8, MaxBytes> bytes{};
    };

    struct Slot {
        Ref ref{};
        u64 tick{};
        u64 device_addr{}; ///< Where `bytes` were last read
        u32 size{};        // 0: nothing streamed yet
        u16 misses{};      ///< Binds in a row whose bytes differed from the slot's last
        u16 direct{};      ///< Binds left that copy without comparing
        u64 hash{};
        u8 history_next{};
        std::array<u8, MaxBytes> bytes{};
        std::array<HistoryEntry, HistoryWays> history{};
    };

    void Reset() noexcept {
        for (auto& stage : slots) {
            for (Slot& slot : stage) {
                slot.size = 0;
                slot.tick = 0;
                slot.misses = 0;
                slot.direct = 0;
                slot.hash = 0;
                slot.history_next = 0;
                for (auto& entry : slot.history) {
                    entry.size = 0;
                    entry.tick = 0;
                    entry.device_addr = 0;
                    entry.hash = 0;
                }
            }
        }
    }

    std::array<std::array<Slot, NumSlots>, NumStages> slots{};
    std::array<u8, MaxBytes> scratch{};
};

template <class Ref>
[[nodiscard]] inline UniformStreamReuse<Ref>& UniformStreamReuseState() noexcept {
    static UniformStreamReuse<Ref> state;
    return state;
}

/// Forgets every slot. Call it when a buffer cache (and so its runtime, scheduler and stream
/// buffer) is created: ticks restart with a new scheduler and the old references are dead.
template <class Runtime>
inline void ResetUniformStreamReuse() noexcept {
    if constexpr (StreamUniformReusable<Runtime>) {
        UniformStreamReuseState<typename Runtime::StreamUniformRef>().Reset();
    }
}

/// Binds the small uniform buffer at `device_addr` for (stage, cbuf `index`) through the stream
/// buffer, reusing the slot's previous allocation when the bytes and the command buffer are the
/// same. Returns false (nothing bound) when the buffer does not qualify, so the caller binds it
/// the classic way.
template <class Runtime, class DeviceMemory>
[[nodiscard]] bool BindSmallUniformReusing(Runtime& runtime, DeviceMemory& device_memory,
                                           size_t stage, u32 index, u32 binding_index,
                                           u64 device_addr, u32 size) {
    using Reuse = UniformStreamReuse<typename Runtime::StreamUniformRef>;
    if (size == 0 || size > Reuse::MaxBytes || stage >= Reuse::NumStages ||
        index >= Reuse::NumSlots) {
        return false;
    }
    Reuse& reuse = UniformStreamReuseState<typename Runtime::StreamUniformRef>();
    typename Reuse::Slot& slot = reuse.slots[stage][index];
    const bool history_enabled = UniformStreamHistoryEnabled();
    slot.device_addr = device_addr;
    if (slot.direct != 0) {
        // A slot whose bytes change every bind: read straight into its bytes, no compare (from
        // skolvolt 9267908a3d). The bytes stay the last bind's, for keeps and their verify.
        --slot.direct;
        device_memory.ReadBlockUnsafe(device_addr, slot.bytes.data(), size);
        slot.size = size;
        if (history_enabled) {
            slot.hash =
                Common::CityHash64(reinterpret_cast<const char*>(slot.bytes.data()), size);
        }
        const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
        std::memcpy(span.data(), slot.bytes.data(), size);
        slot.ref = runtime.LastStreamUniformRef();
        slot.tick = runtime.CurrentTick();
        return true;
    }
    // Guest memory in one piece of host memory (most of these buffers lie within a page) is
    // compared where it is; other ranges are gathered into scratch first.
    const u8* guest = nullptr;
    if constexpr (requires { device_memory.GetSpan(device_addr, size_t{size}); }) {
        guest = device_memory.GetSpan(device_addr, size_t{size});
    }
    if (guest == nullptr) {
        device_memory.ReadBlockUnsafe(device_addr, reuse.scratch.data(), size);
        guest = reuse.scratch.data();
    }
    const u64 current_tick = runtime.CurrentTick();

    if (!history_enabled) {
        const bool same_bytes =
            slot.size == size && std::memcmp(slot.bytes.data(), guest, size) == 0;
        if (same_bytes && slot.tick == current_tick) {
            slot.misses = 0;
            runtime.RebindStreamUniform(slot.ref, size);
            return true;
        }
        if (!same_bytes) {
            // The slot takes the bytes first and the stream gets the slot's: the guest may write
            // its memory meanwhile, and the slot must hold what its allocation holds.
            slot.size = size;
            std::memcpy(slot.bytes.data(), guest, size);
        }
        const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
        std::memcpy(span.data(), slot.bytes.data(), size);
        slot.ref = runtime.LastStreamUniformRef();
        // After the allocation: the tick the stream region is tagged with is the one now.
        slot.tick = current_tick;
        if (same_bytes) {
            slot.misses = 0;
        } else if (++slot.misses >= Reuse::MissStreakSingle) {
            slot.misses = 0;
            slot.direct = Reuse::DirectBindsSingle;
        }
        return true;
    }

    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(guest), size);
    // 1. The bind before this one: same range, same bytes, same command buffer.
    const bool same_bytes = slot.size == size && slot.device_addr == device_addr &&
                            slot.hash == hash &&
                            std::memcmp(slot.bytes.data(), guest, size) == 0;
    if (same_bytes && slot.tick == current_tick) {
        slot.misses = 0;
        runtime.RebindStreamUniform(slot.ref, size);
        return true;
    }
    // 2. The slot's history: actors that alternate their constant buffers every draw.
    for (auto& entry : slot.history) {
        if (entry.size != size || entry.device_addr != device_addr || entry.tick != current_tick) {
            continue;
        }
        if (entry.hash == hash && std::memcmp(entry.bytes.data(), guest, size) == 0) {
            slot.misses = 0;
            slot.ref = entry.ref;
            slot.tick = current_tick;
            slot.size = size;
            slot.hash = hash;
            std::memcpy(slot.bytes.data(), guest, size);
            runtime.RebindStreamUniform(entry.ref, size);
            return true;
        }
        // Same range, other bytes: this entry takes the new allocation directly.
        const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
        std::memcpy(span.data(), guest, size);
        const auto ref = runtime.LastStreamUniformRef();
        slot.ref = ref;
        slot.tick = current_tick;
        slot.size = size;
        slot.hash = hash;
        std::memcpy(slot.bytes.data(), guest, size);
        entry.ref = ref;
        entry.tick = current_tick;
        entry.hash = hash;
        std::memcpy(entry.bytes.data(), guest, size);
        if (++slot.misses >= Reuse::MissStreak) {
            slot.misses = 0;
            slot.direct = Reuse::DirectBinds;
        }
        return true;
    }
    // 3. A range this slot has not streamed in this command buffer: allocate and remember it.
    const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
    std::memcpy(span.data(), guest, size);
    const auto ref = runtime.LastStreamUniformRef();
    slot.ref = ref;
    slot.tick = current_tick;
    slot.size = size;
    slot.hash = hash;
    std::memcpy(slot.bytes.data(), guest, size);
    auto& entry = slot.history[slot.history_next];
    slot.history_next = static_cast<u8>((slot.history_next + 1) % Reuse::HistoryWays);
    entry.ref = ref;
    entry.tick = current_tick;
    entry.device_addr = device_addr;
    entry.size = size;
    entry.hash = hash;
    std::memcpy(entry.bytes.data(), guest, size);
    if (++slot.misses >= Reuse::MissStreak) {
        slot.misses = 0;
        slot.direct = Reuse::DirectBinds;
    }
    return true;
}

/// Whether RebindKeptUniform binds the slot's last allocation again for this range: the slot's
/// last bind read the same range in the command buffer that is current now.
template <class Runtime>
[[nodiscard]] bool KeptUniformMatches(Runtime& runtime, size_t stage, u32 index, u64 device_addr,
                                      u32 size) {
    using Reuse = UniformStreamReuse<typename Runtime::StreamUniformRef>;
    if (size == 0 || size > Reuse::MaxBytes || stage >= Reuse::NumStages ||
        index >= Reuse::NumSlots) {
        return false;
    }
    const typename Reuse::Slot& slot =
        UniformStreamReuseState<typename Runtime::StreamUniformRef>().slots[stage][index];
    return slot.size == size && slot.device_addr == device_addr &&
           slot.tick == runtime.CurrentTick();
}

/// Uniform keeps: binds the slot's last allocation again without reading guest memory,
/// when that bind read the same range in this command buffer. The caller guarantees that nothing
/// ordered before this draw wrote the range since that bind (the previous draw's). Returns false
/// (nothing bound) otherwise.
template <class Runtime>
[[nodiscard]] bool RebindKeptUniform(Runtime& runtime, size_t stage, u32 index, u64 device_addr,
                                     u32 size) {
    if (!KeptUniformMatches(runtime, stage, index, device_addr, size)) {
        return false;
    }
    using Reuse = UniformStreamReuse<typename Runtime::StreamUniformRef>;
    const typename Reuse::Slot& slot =
        UniformStreamReuseState<typename Runtime::StreamUniformRef>().slots[stage][index];
    runtime.RebindStreamUniform(slot.ref, size);
    return true;
}

/// EDEN_NX2_UBO_KEEP_VERIFY: whether guest memory still holds the bytes the slot's last
/// bind read from the range RebindKeptUniform would reuse.
template <class Runtime, class DeviceMemory>
[[nodiscard]] bool KeptUniformHolds(DeviceMemory& device_memory, size_t stage, u32 index,
                                    u64 device_addr, u32 size) {
    using Reuse = UniformStreamReuse<typename Runtime::StreamUniformRef>;
    if (size == 0 || size > Reuse::MaxBytes || stage >= Reuse::NumStages ||
        index >= Reuse::NumSlots) {
        return false;
    }
    Reuse& reuse = UniformStreamReuseState<typename Runtime::StreamUniformRef>();
    const typename Reuse::Slot& slot = reuse.slots[stage][index];
    if (slot.size != size || slot.device_addr != device_addr) {
        return false;
    }
    device_memory.ReadBlockUnsafe(device_addr, reuse.scratch.data(), size);
    return std::memcmp(slot.bytes.data(), reuse.scratch.data(), size) == 0;
}

} // namespace VideoCommon
