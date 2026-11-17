// SPDX-License-Identifier: GPL-3.0-or-later
// Stall meter: time guest memory faults cost the threads that took them, summed per kind and
// printed after each EDEN_FPS line (headless/graphics.cpp) as EDEN_STALL. The counters live in
// src/memory_pages.cpp, which is part of the common library.
#pragma once

#include <chrono>
#include <cstdint>

// Guest memory work done on the thread that touched the memory (src/guest_memory_service.inc).
// The GPU thread's own causes are EDEN_NX2_STUTTER_TRACE's (video_core/nx2_stutter_trace.h).
enum EdenStallKind : int {
    EdenStallFault = 0,     // HandleGuestWriteFault: a whole fault, everything below included
    EdenStallUnpack = 1,    // UnpackSlot: a slot CRAM packed, unpacked
    EdenStallSwapIn = 2,    // SwapIn: a slot read back from the swap file
    EdenStallRecover = 3,   // RecoverForFault: other owners asked to give memory back
    EdenStallBlockWait = 4, // WaitForBlock: a write waiting for a free block
    EdenStallKinds = 5,
};

extern "C" void eden_stall_add(int kind, std::uint64_t ns) noexcept;
// Sums since the last call, then resets them: total ns, count and longest ns per kind.
extern "C" void eden_stall_take(std::uint64_t* total_ns, std::uint64_t* count,
                                std::uint64_t* longest_ns) noexcept;

class EdenStallScope {
public:
    explicit EdenStallScope(int kind) noexcept
        : kind_{kind}, start_{std::chrono::steady_clock::now()} {}
    ~EdenStallScope() {
        eden_stall_add(kind_, static_cast<std::uint64_t>(
                                  std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - start_)
                                      .count()));
    }
    EdenStallScope(const EdenStallScope&) = delete;
    EdenStallScope& operator=(const EdenStallScope&) = delete;

private:
    int kind_;
    std::chrono::steady_clock::time_point start_;
};
