// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <sys/mman.h>
#include <unistd.h>
#include <xbyak/xbyak.h>
namespace Common {
void* AllocateMemoryPages(std::size_t) noexcept;
void FreeMemoryPages(void*) noexcept;
void* MapExecutableAlias(void*, std::size_t) noexcept;
// A cache that takes its memory as it fills (src/memory_pages.cpp).
void* ReserveCodeRange(std::size_t, void** executable) noexcept;
bool CommitCodeRange(const void* executable, std::size_t bytes) noexcept;
void DecommitCodeRange(const void* executable, std::size_t keep_bytes) noexcept;
bool FreeCodeRange(void* executable) noexcept;
}
// The guest memory service's reclaim path (src/guest_memory_service.inc): trims the driver's
// unbound granules and packs cold guest memory to give the pool room. Null outside the app.
extern "C" [[gnu::weak]] std::size_t eden_memory_recover(std::size_t bytes) noexcept;
// Separate RW/NX and RX views avoid changing page permissions per compiled block.
// Unsupported native JIT allocation falls back to the qualified owned W^X path.
inline Xbyak::Allocator* EdenJitAllocator() {
    struct Allocator final : Xbyak::Allocator {
        struct Mapping { void* writable; std::size_t size; bool grows; };
        std::mutex mutex;
        std::unordered_map<std::uint8_t*, Mapping> mappings;
        std::uint8_t* alloc(std::size_t size) override {
#ifdef EDEN_JIT_ALIAS_NATIVE
            auto* diagnostics = stdout; // Native stdout is the separate heap log.
#else
            auto* diagnostics = stderr; // Host stdout is the strict lifecycle receipt.
#endif
            const long page = sysconf(_SC_PAGESIZE);
            if (!size || page <= 0 || size > std::numeric_limits<std::size_t>::max() - page)
                return nullptr;
            const auto span = (size + page - 1) & ~(std::size_t(page) - 1);
            void* executable = MAP_FAILED;
            void* writable = MAP_FAILED;
            // A large cache is address space, and memory comes behind it as the JIT fills it
            // (commit below). Its first step now: the code generator writes at once.
            void* grown_executable = nullptr;
            if (void* const grown = Common::ReserveCodeRange(size, &grown_executable)) {
                bool kept = false;
                if (Common::CommitCodeRange(grown_executable, 1)) {
                    try {
                        std::lock_guard lock(mutex);
                        mappings.emplace(static_cast<std::uint8_t*>(grown_executable), Mapping{grown, size, true});
                        kept = true;
                    } catch (...) {
                    }
                }
                if (kept) {
                    std::fprintf(diagnostics, "EDEN_JIT_ALIAS bytes=%zu active=1 code=%p grows=1\n", size,
                                 grown_executable);
                    return static_cast<std::uint8_t*>(grown_executable);
                }
                if (!Common::FreeCodeRange(grown_executable)) std::abort();
            }
#ifdef EDEN_JIT_ALIAS_NATIVE
            writable = Common::AllocateMemoryPages(size);
            if (!writable) return nullptr;
            executable = Common::MapExecutableAlias(writable, size);
            if (!executable) {
                std::fprintf(diagnostics, "EDEN_JIT_ALIAS bytes=%zu active=0 errno=%d\n", span, errno);
                return static_cast<std::uint8_t*>(writable);
            }
#else
            const int executable_fd = memfd_create("eden-jit", MFD_CLOEXEC);
            if (executable_fd >= 0 && ftruncate(executable_fd, span) == 0) {
                executable = mmap(nullptr, span, PROT_READ | PROT_EXEC, MAP_SHARED, executable_fd, 0);
                writable = mmap(nullptr, span, PROT_READ | PROT_WRITE, MAP_SHARED, executable_fd, 0);
            }
            if (executable_fd >= 0 && close(executable_fd) != 0) std::abort();
#endif
            if (executable != MAP_FAILED && executable && writable != MAP_FAILED && writable && executable != writable) {
                auto* pointer = static_cast<std::uint8_t*>(executable);
                try {
                    std::lock_guard lock(mutex);
                    mappings.emplace(pointer, Mapping{writable, span, false});
                    std::fprintf(diagnostics, "EDEN_JIT_ALIAS bytes=%zu active=1\n", span);
                    return pointer;
                } catch (...) {
                    // Release both views before reporting allocation failure.
                    if (munmap(executable, span) != 0) std::abort();
#ifdef EDEN_JIT_ALIAS_NATIVE
                    Common::FreeMemoryPages(writable);
#else
                    if (munmap(writable, span) != 0) std::abort();
#endif
                    return nullptr;
                }
            }
#ifdef EDEN_JIT_ALIAS_NATIVE
            std::abort(); // Distinct views are required by the direct mapper.
#else
            if (executable != MAP_FAILED && executable && munmap(executable, span) != 0) std::abort();
            if (writable != MAP_FAILED && writable && writable != executable && munmap(writable, span) != 0) std::abort();
            std::fprintf(diagnostics, "EDEN_JIT_ALIAS bytes=%zu active=0 errno=%d\n", span, errno);
            return static_cast<std::uint8_t*>(Common::AllocateMemoryPages(size));
#endif
        }
        std::uint8_t* writableAddress(std::uint8_t* pointer) override {
            std::lock_guard lock(mutex);
            const auto entry = mappings.find(pointer);
            return entry == mappings.end() ? pointer : static_cast<std::uint8_t*>(entry->second.writable);
        }
        // Memory behind the first `bytes` of a cache: asked before the JIT writes that far.
        void commit(std::uint8_t* pointer, std::size_t bytes) override {
            if (Common::CommitCodeRange(pointer, bytes)) return;
            // CommitCodeRange tried its own three recoveries and found no step. The load that took
            // the pool's last contiguous pieces releases them over the next moments, and the
            // swapper needs a tick to pack cold guest memory, so wait a bounded while and ask
            // again before the cache cannot go on. One line per episode.
            constexpr auto wait_limit = std::chrono::seconds(3);
            constexpr auto pause = std::chrono::milliseconds(40);
            const auto begin = std::chrono::steady_clock::now();
            int tries = 0;
            bool ok = false;
            while (&eden_memory_recover != nullptr && !ok &&
                   std::chrono::steady_clock::now() - begin < wait_limit) {
                std::this_thread::sleep_for(pause);
                ++tries;
                (void)eden_memory_recover(std::size_t{32} << 20);
                ok = Common::CommitCodeRange(pointer, bytes);
            }
            const auto waited =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin);
            std::fprintf(stderr, "EDEN_JIT_COMMIT_WAIT tries=%d waited_ms=%lld ok=%d\n", tries,
                         static_cast<long long>(waited.count()), ok ? 1 : 0);
            if (ok) return;
            std::fputs("JIT code cache: the console has no memory for its next part\n", stderr);
            std::abort();
        }
        void free(std::uint8_t* pointer) override {
            {
                std::lock_guard lock(mutex);
                const auto entry = mappings.find(pointer);
                if (entry != mappings.end()) {
                    const auto mapping = entry->second;
                    if (mapping.grows) {
                        if (!Common::FreeCodeRange(pointer)) std::abort();
                        mappings.erase(entry);
                        return;
                    }
                    if (munmap(pointer, mapping.size) != 0) std::abort();
#ifdef EDEN_JIT_ALIAS_NATIVE
                    Common::FreeMemoryPages(mapping.writable);
#else
                    if (munmap(mapping.writable, mapping.size) != 0) std::abort();
#endif
                    mappings.erase(entry);
                    return;
                }
            }
            Common::FreeMemoryPages(pointer);
        }
        bool useProtect() const override { return false; }
    };
    static Allocator allocator;
    return &allocator;
}
