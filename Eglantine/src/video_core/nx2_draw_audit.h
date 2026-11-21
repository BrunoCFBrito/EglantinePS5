// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
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

// Draws grouped by the render pass target they open: colour target 0 (address and format), the
// depth target (address and format), the colour target count, whether the pass is depth-only or
// has no target at all, and the target size. A pass family's share of the draw stream sizes what
// a per-pass lever could win: depth-only passes are the sun shadow map, small targets the water
// ripple, reflections and effects.
struct PassKey {
    u64 color; ///< rt[0] address, its low bits moved out by |format; 0 with no colour target
    u64 depth; ///< zeta address, its low bits moved out by |format; 0 with depth off
    u32 rt_count;
    u32 width;
    u32 height;
    u32 flags; ///< bit 0 depth-only, bit 1 no target

    bool operator==(const PassKey&) const = default;
};

struct PassGroup {
    PassKey key{};
    u64 draws{};
};

inline constexpr size_t PassTableSize = 256;
inline constexpr size_t PassProbes = 8;
inline std::array<PassGroup, PassTableSize> g_passes{};
inline u64 g_pass_total{};
inline u64 g_pass_depth_only{};
inline u64 g_pass_shadow{};
inline u64 g_pass_small{};
inline u64 g_pass_overflow{};

[[nodiscard]] inline u64 HashPass(const PassKey& key) {
    u64 hash = key.color ^ (key.depth * 0x9e3779b97f4a7c15ULL);
    hash ^= (u64{key.rt_count} << 32) ^ (u64{key.width} << 11) ^ u64{key.height};
    hash ^= u64{key.flags} * 0xc2b2ae3d27d4eb4fULL;
    return hash ^ (hash >> 29);
}

/// Counts one draw in its pass group. Cheap: one hash, then a probe over groups already seen.
inline void CountPass(const PassKey& key) {
    if (!g_enabled) {
        return;
    }
    ++g_pass_total;
    if (key.flags & 1) {
        ++g_pass_depth_only;
        g_pass_shadow += (key.width >= 4096 || key.height >= 4096) ? 1 : 0;
    }
    g_pass_small += (key.width <= 512 && key.height <= 512) ? 1 : 0;
    size_t index = HashPass(key) & (PassTableSize - 1);
    for (size_t probe = 0; probe < PassProbes; ++probe) {
        PassGroup& group = g_passes[index];
        if (group.draws == 0) {
            group.key = key;
            group.draws = 1;
            return;
        }
        if (group.key == key) {
            ++group.draws;
            return;
        }
        index = (index + 1) & (PassTableSize - 1);
    }
    ++g_pass_overflow;
}

inline void LogPasses(double seconds) {
    const auto rate = [seconds](u64 value) { return static_cast<double>(value) / seconds; };
    const auto share = [](u64 part, u64 total) {
        return total == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(total);
    };
    std::array<PassGroup, PassTableSize> groups{};
    size_t count = 0;
    for (const PassGroup& group : g_passes) {
        if (group.draws != 0) {
            groups[count++] = group;
        }
    }
    std::partial_sort(groups.begin(), groups.begin() + std::min<size_t>(count, 16), groups.begin() + count,
                      [](const PassGroup& a, const PassGroup& b) { return a.draws > b.draws; });
    LOG_INFO(HW_GPU,
             "Draw pass audit per second: {:.0f} draws in {} groups, depth-only {:.0f} "
             "({:.1f}%), shadow {:.0f} ({:.1f}%), small {:.0f} ({:.1f}%), ungrouped {:.0f}",
             rate(g_pass_total), count, rate(g_pass_depth_only), share(g_pass_depth_only, g_pass_total),
             rate(g_pass_shadow), share(g_pass_shadow, g_pass_total), rate(g_pass_small),
             share(g_pass_small, g_pass_total), rate(g_pass_overflow));
    for (size_t i = 0; i < std::min<size_t>(count, 15); ++i) {
        const PassGroup& group = groups[i];
        LOG_INFO(HW_GPU,
                 "Draw pass {}: {:.0f}/s ({:.1f}%) color0={:#x}/{:#x} depth={:#x}/{:#x} rt={} {}x{}{}",
                 i + 1, rate(group.draws), share(group.draws, g_pass_total),
                 group.key.color >> 8, group.key.color & 0xff, group.key.depth >> 8,
                 group.key.depth & 0xff, group.key.rt_count, group.key.width, group.key.height,
                 (group.key.flags & 1) ? " depth-only" : (group.key.flags & 2) ? " no-target" : "");
    }
    g_passes = {};
    g_pass_total = 0;
    g_pass_depth_only = 0;
    g_pass_shadow = 0;
    g_pass_small = 0;
    g_pass_overflow = 0;
}

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
    LogPasses(seconds);
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
