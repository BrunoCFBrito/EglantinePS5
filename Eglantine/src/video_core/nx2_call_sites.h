// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>

#include "common/common_types.h"

// Names for the deferred calls on the split's back that end fast draw reuse, so each can be
// judged (a call that cannot change what a draw resolves to belongs behind
// GpuSplit::FastDrawNeutral). The site rides in the break's method field; the fast draw report
// logs, every 10 s, which call ended the chain and how many draws it cost.
namespace VideoCommon::Nx2CallSite {

enum Site : u32 {
    Unlabelled,
    PullerFenceRelease,
    PullerSyncpoint,
    PullerReport,
    PullerSemaphoreRelease,
    PullerReference,
    PullerIdle,
    PullerInvalidateCache,
    GpuThreadFlushRegion,
    GpuThreadInvalidateRegion,
    GpuThreadWait,
    DmaPusherFlushCaching,
    DmaPusherEngineState,
    DmaPusherMemory,
    DmaPusherListEnd,
    SchedulerBindChannel,
    DeferredFunction,
    PullerReadFlush,
    DmaPusherGpSyncFence,
    DmaPusherListFlushPages,
    DmaPusherListFlush,
    MacroDmaFlush,
    NumSites, ///< Number of sites.
};

inline constexpr std::array<const char*, NumSites> names{
    "unlabelled or other", "puller fence release", "puller syncpoint",
    "puller report",  "puller semaphore",     "puller reference",
    "puller idle",    "puller invalidate",    "thread flush region",
    "thread invalidate region", "thread wait", "dma flush caching",
    "dma engine state", "dma memory",          "dma list end",
    "scheduler bind channel", "deferred function", "puller read flush",
    "dma gp sync fence", "dma list flush pages", "dma list flush", "macro dma flush",
};

} // namespace VideoCommon::Nx2CallSite
