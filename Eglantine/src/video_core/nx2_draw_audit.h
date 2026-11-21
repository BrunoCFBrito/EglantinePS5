// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <cstdlib>

#include "common/common_types.h"
#include "common/logging.h"

// EDEN_NX2_DRAW_AUDIT: what the guest's draw stream is made of, so a draw-count lever can be
// sized before it is designed. Every 10 s it logs, per second: how many draws are empty
// (nothing to render), suppressed (ShouldExecute false), identical to the draw before, or open a
// new pipeline run (the vertex or pixel program changed); the element-count distribution; and
// the index formats. `runs` is the least number of draws a batching scheme could leave.
namespace VideoCommon::Nx2DrawAudit {

inline bool g_enabled{};

struct Signature {
    u64 vs;
    u64 ps;
    u64 first;
    u64 count;
    u64 topology;
    u64 index_address;
    u64 index_format;

    bool operator==(const Signature&) const = default;
};

struct Counters {
    u64 draws{};
    u64 empty{};
    u64 suppressed{};
    u64 repeated{};   ///< Equal to the draw before, empty draws included.
    u64 runs{};       ///< A new vertex or pixel program: a batch has to end here.
    u64 instanced{};  ///< instance_count > 1.
    u64 indirect{};   ///< Counted apart: their counts live in guest memory.
    u64 indexed[3]{}; ///< u8, u16, u32.
    /// Element counts: 0, 1..8, 9..32, 33..128, 129..1024, >1024.
    std::array<u64, 6> elements{};
    bool have_previous{};
    Signature previous{};
    std::chrono::steady_clock::time_point window{std::chrono::steady_clock::now()};
};

inline Counters g_counters;

inline void Count(bool indexed, u32 element_count, u32 instance_count, u32 index_format,
                  bool suppressed, u64 vs, u64 ps, u64 first, u64 topology, u64 index_address) {
    if (!g_enabled) {
        return;
    }
    auto& c = g_counters;
    ++c.draws;
    const Signature signature{vs, ps, first, element_count, topology, index_address, index_format};
    c.repeated += (c.have_previous && signature == c.previous) ? 1 : 0;
    c.runs += (!c.have_previous || signature.vs != c.previous.vs ||
               signature.ps != c.previous.ps)
                  ? 1
                  : 0;
    c.previous = signature;
    c.have_previous = true;
    c.empty += (element_count == 0 || instance_count == 0) ? 1 : 0;
    c.suppressed += suppressed ? 1 : 0;
    c.instanced += instance_count > 1 ? 1 : 0;
    if (indexed && index_format < 3) {
        ++c.indexed[index_format];
    }
    const size_t bucket = element_count == 0      ? 0
                          : element_count <= 8    ? 1
                          : element_count <= 32   ? 2
                          : element_count <= 128  ? 3
                          : element_count <= 1024 ? 4
                                                  : 5;
    ++c.elements[bucket];
    if ((c.draws & 2047) != 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const double seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(now - c.window).count();
    if (seconds < 10.0) {
        return;
    }
    const auto rate = [seconds](u64 value) { return static_cast<double>(value) / seconds; };
    LOG_INFO(HW_GPU,
             "Draw audit per second: {:.0f} draws, {:.0f} empty, {:.0f} suppressed, {:.0f} "
             "repeat the draw before, {:.0f} open a new program run, {:.0f} instanced, {:.0f} "
             "indirect | elements "
             "0:{:.0f} 1-8:{:.0f} 9-32:{:.0f} 33-128:{:.0f} 129-1024:{:.0f} 1025+:{:.0f} | index "
             "u8:{:.0f} u16:{:.0f} u32:{:.0f}",
             rate(c.draws), rate(c.empty), rate(c.suppressed), rate(c.repeated), rate(c.runs),
             rate(c.instanced), rate(c.indirect), rate(c.elements[0]), rate(c.elements[1]),
             rate(c.elements[2]), rate(c.elements[3]), rate(c.elements[4]), rate(c.elements[5]),
             rate(c.indexed[0]), rate(c.indexed[1]), rate(c.indexed[2]));
    c = Counters{.window = now};
}

inline void SetEnabledFromEnv() {
    const char* const value = std::getenv("EDEN_NX2_DRAW_AUDIT");
    g_enabled = value != nullptr && value[0] == '1';
}

} // namespace VideoCommon::Nx2DrawAudit
