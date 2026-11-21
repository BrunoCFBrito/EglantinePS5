// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// EDEN_NX2_EMUTLS_STATS: the PS5 build links compiler-rt's emulated TLS (every unit is built with
// -femulated-tls), so one thread_local read is one __emutls_get_address call. This file wraps that
// call (--wrap=__emutls_get_address) and, when the switch is set, counts the calls per thread and
// per return address and prints an EDEN_EMUTLS line set to stderr every 10 s. Thread names come
// from Performance::RegisterWorker. Both the wrapper and this file are build-time opt-in
// (EDEN_EMUTLS_STATS_BUILD=1, headless/CMakeLists.txt): with the switch off the wrapper is one
// relaxed load and a branch, then the real call, which is still too much for every thread_local
// read in a default build. No thread_local is used here on purpose: any thread_local read in the
// wrapper would call the wrapper.
#include "nx2_emutls_stats.h"

#if defined(EDEN_EMUTLS_STATS)

#include <pthread.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" void* __real___emutls_get_address(void* object);

namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t ThreadSlots = 256;
constexpr size_t ThreadProbes = 8;
constexpr size_t SiteSlots = 512;
constexpr size_t SiteProbes = 4;
constexpr size_t ReportedSites = 12;
constexpr double ReportSeconds = 10.0;

struct ThreadSlot {
    std::atomic<uint64_t> id{0};
    std::atomic<uint64_t> calls{0};
    std::atomic<unsigned> busy{0}; ///< Set while this thread is inside the wrapper (recursion guard)
    uint64_t last_calls{};
    char name[24]{};
};

struct SiteEntry {
    std::atomic<uint64_t> pc{0};
    std::atomic<uint64_t> count{0};
    uint64_t last_count{};
};

ThreadSlot g_threads[ThreadSlots];
SiteEntry g_sites[SiteSlots];
std::atomic<uint64_t> g_total{0};
std::atomic<bool> g_enabled{false};
std::atomic<bool> g_initialised{false};
std::atomic<bool> g_reporting{false};
std::atomic<uint64_t> g_anchor{0};
std::atomic<bool> g_threads_full{false};
std::atomic<bool> g_window_set{false};
Clock::time_point g_window{};
/// g_total at the previous report: the per-window rate is a delta, g_total itself never resets.
uint64_t g_last_total{};
/// g_total of the window just reported, the denominator of every share.
uint64_t g_last_window_total{};

/// Runtime address of a known function, printed once so the site PCs can be turned into eboot
/// offsets (its static address comes from nm on build/headless-native/eboot.elf).
extern "C" __attribute__((noinline, used)) void EdenNx2EmutlsStatsAnchor() {}

[[nodiscard]] uint64_t CurrentThreadId() {
#if defined(__aarch64__)
    return reinterpret_cast<uint64_t>(__builtin_thread_pointer());
#else
    // PS5 is x86-64 and the thread pointer register is not set up (TLS is emulated), so
    // __builtin_thread_pointer() would read %fs:0. pthread_self is the thread's control block
    // and is unique per host thread.
    return reinterpret_cast<uint64_t>(pthread_self());
#endif
}

void Initialise() {
    // Set first: getenv is called with the wrapper already on the stack, and this keeps a nested
    // call from initialising again.
    g_initialised.store(true, std::memory_order_relaxed);
    const char* const value = std::getenv("EDEN_NX2_EMUTLS_STATS");
    g_enabled.store(value != nullptr && value[0] == '1', std::memory_order_relaxed);
    g_anchor.store(reinterpret_cast<uint64_t>(&EdenNx2EmutlsStatsAnchor), std::memory_order_relaxed);
}

inline void EnsureInitialised() {
    if (!g_initialised.load(std::memory_order_relaxed)) [[unlikely]] {
        Initialise();
    }
}

[[nodiscard]] ThreadSlot* FindSlot(uint64_t id) {
    // Open addressing on the thread's control block. A linear scan would cost more than the TLS
    // call it measures at ten million calls a second.
    size_t index = (id >> 4) & (ThreadSlots - 1);
    for (size_t probe = 0; probe < ThreadProbes; ++probe) {
        ThreadSlot& slot = g_threads[index];
        const uint64_t slot_id = slot.id.load(std::memory_order_relaxed);
        if (slot_id == id) {
            return &slot;
        }
        if (slot_id == 0) {
            uint64_t expected = 0;
            if (slot.id.compare_exchange_strong(expected, id, std::memory_order_relaxed)) {
                return &slot;
            }
            if (expected == id) {
                return &slot;
            }
        }
        index = (index + 1) & (ThreadSlots - 1);
    }
    g_threads_full.store(true, std::memory_order_relaxed);
    return &g_threads[ThreadSlots - 1]; // Every slot taken: the last one merges the rest.
}

inline void CountSite(uint64_t pc) {
    size_t index = (pc >> 2) % SiteSlots;
    for (size_t probe = 0; probe < SiteProbes; ++probe) {
        SiteEntry& entry = g_sites[index];
        const uint64_t entry_pc = entry.pc.load(std::memory_order_relaxed);
        if (entry_pc == pc) {
            entry.count.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (entry_pc == 0) {
            uint64_t expected = 0;
            if (entry.pc.compare_exchange_strong(expected, pc, std::memory_order_relaxed)) {
                entry.count.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            if (expected == pc) {
                entry.count.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
        index = (index + 1) % SiteSlots;
    }
}

void Report(Clock::time_point now) {
    if (g_reporting.exchange(true, std::memory_order_relaxed)) {
        return;
    }
    const double seconds = std::chrono::duration<double>(now - g_window).count();
    if (seconds < ReportSeconds) {
        g_reporting.store(false, std::memory_order_relaxed);
        return;
    }
    g_window = now;
    const uint64_t total = g_total.load(std::memory_order_relaxed);
    const uint64_t window_total = total - g_last_total;
    g_last_total = total;
    g_last_window_total = window_total;
    std::fprintf(stderr,
                 "EDEN_EMUTLS anchor=0x%llx window_s=%.2f calls_s=%.0f window=%llu total=%llu "
                 "threads_full=%d\n",
                 static_cast<unsigned long long>(g_anchor.load(std::memory_order_relaxed)),
                 seconds, static_cast<double>(window_total) / seconds,
                 static_cast<unsigned long long>(window_total),
                 static_cast<unsigned long long>(total),
                 g_threads_full.load(std::memory_order_relaxed) ? 1 : 0);
    for (size_t i = 0; i < ThreadSlots; ++i) {
        ThreadSlot& slot = g_threads[i];
        if (slot.id.load(std::memory_order_relaxed) == 0) {
            continue;
        }
        const uint64_t calls = slot.calls.load(std::memory_order_relaxed);
        const uint64_t delta = calls - slot.last_calls;
        slot.last_calls = calls;
        if (delta == 0) {
            continue;
        }
        std::fprintf(stderr,
                     "EDEN_EMUTLS thread slot=%zu name=%s id=0x%llx calls_s=%.0f total=%llu\n",
                     i, slot.name[0] != '\0' ? slot.name : "?",
                     static_cast<unsigned long long>(slot.id.load(std::memory_order_relaxed)),
                     static_cast<double>(delta) / seconds,
                     static_cast<unsigned long long>(calls));
    }
    // The busiest call sites of the window.
    uint64_t top_pc[ReportedSites]{};
    uint64_t top_rate[ReportedSites]{};
    for (SiteEntry& entry : g_sites) {
        const uint64_t pc = entry.pc.load(std::memory_order_relaxed);
        if (pc == 0) {
            continue;
        }
        const uint64_t count = entry.count.load(std::memory_order_relaxed);
        const uint64_t delta = count - entry.last_count;
        entry.last_count = count;
        for (size_t i = 0; i < ReportedSites; ++i) {
            if (delta > top_rate[i]) {
                for (size_t j = ReportedSites - 1; j > i; --j) {
                    top_rate[j] = top_rate[j - 1];
                    top_pc[j] = top_pc[j - 1];
                }
                top_rate[i] = delta;
                top_pc[i] = pc;
                break;
            }
        }
    }
    for (size_t i = 0; i < ReportedSites; ++i) {
        if (top_rate[i] == 0) {
            break;
        }
        const double share = g_last_window_total == 0 ? 0.0
                              : 100.0 * static_cast<double>(top_rate[i]) /
                                    static_cast<double>(g_last_window_total);
        std::fprintf(stderr, "EDEN_EMUTLS site %zu pc=0x%llx calls_s=%.0f share=%.1f%%\n", i + 1,
                     static_cast<unsigned long long>(top_pc[i]),
                     static_cast<double>(top_rate[i]) / seconds, share);
    }
    g_reporting.store(false, std::memory_order_relaxed);
}

} // namespace

namespace Eden::Nx2EmutlsStats {

void NoteCurrentThread(const char* name) {
    EnsureInitialised();
    if (name == nullptr) {
        return;
    }
    ThreadSlot* const slot = FindSlot(CurrentThreadId());
    std::strncpy(slot->name, name, sizeof(slot->name) - 1);
    slot->name[sizeof(slot->name) - 1] = '\0';
}

} // namespace Eden::Nx2EmutlsStats

extern "C" void* __wrap___emutls_get_address(void* object) {
    if (!g_enabled.load(std::memory_order_relaxed)) {
        EnsureInitialised();
        if (!g_enabled.load(std::memory_order_relaxed)) {
            return __real___emutls_get_address(object);
        }
    }
    ThreadSlot* const slot = FindSlot(CurrentThreadId());
    if (slot->busy.load(std::memory_order_relaxed) != 0) {
        // A call made by something this wrapper itself called (a clock read, the report's
        // fprintf): count it as the real function, never recurse.
        return __real___emutls_get_address(object);
    }
    slot->busy.store(1, std::memory_order_relaxed);
    slot->calls.fetch_add(1, std::memory_order_relaxed);
    const uint64_t total = g_total.fetch_add(1, std::memory_order_relaxed) + 1;
    CountSite(reinterpret_cast<uint64_t>(__builtin_return_address(0)));
    if ((total & 4095) == 0) {
        const auto now = Clock::now();
        if (g_window_set.exchange(true, std::memory_order_relaxed)) {
            Report(now);
        } else {
            g_window = now;
        }
    }
    void* const result = __real___emutls_get_address(object);
    slot->busy.store(0, std::memory_order_relaxed);
    return result;
}

#endif // defined(EDEN_EMUTLS_STATS)
