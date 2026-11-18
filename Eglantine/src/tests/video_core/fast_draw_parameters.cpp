// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/engines/maxwell_3d.h"
#include "video_core/nx2_fast_draw.h"

TEST_CASE("A vertex_id_base write keeps the fast draw chain", "[video_core][fast_draw]") {
    // OoT rewrites 0x1118 from its draw macros on about half its draws (6,400/s measured on the
    // console). macro.cpp is the only writer and nothing reads the register, so it cannot change
    // what a resolve produces; treating it as a light change keeps the delta path for those draws.
    REQUIRE(Tegra::Engines::Maxwell3D::IsFastDrawParameter(MAXWELL3D_REG_INDEX(vertex_id_base)));
}

TEST_CASE("A vertex stream stride write ends the chain unless it is dynamic state",
          "[video_core][fast_draw]") {
    // Strides feed the pipeline key and the repack mask, so a kept pipeline would be wrong: the
    // stride word stays a break until the pipeline cache reports dynamic vertex input with
    // extended dynamic state (Lichii #2116 / EglantinePS5 #129).
    const bool previous = VideoCommon::Nx2FastDraw::g_dynamic_vertex_streams.exchange(false);
    REQUIRE_FALSE(
        Tegra::Engines::Maxwell3D::IsFastDrawParameter(MAXWELL3D_REG_INDEX(vertex_streams)));
    VideoCommon::Nx2FastDraw::g_dynamic_vertex_streams.store(true);
    REQUIRE(Tegra::Engines::Maxwell3D::IsFastDrawParameter(MAXWELL3D_REG_INDEX(vertex_streams)));
    VideoCommon::Nx2FastDraw::g_dynamic_vertex_streams.store(previous);
}

TEST_CASE("A counter reset or render enable write keeps the fast draw chain",
          "[video_core][fast_draw]") {
    // Executable methods, so the method guard used to break on every write (Lichii #2118).
    REQUIRE(Tegra::Engines::Maxwell3D::IsFastDrawParameter(MAXWELL3D_REG_INDEX(clear_report_value)));
    REQUIRE(Tegra::Engines::Maxwell3D::IsFastDrawParameter(MAXWELL3D_REG_INDEX(render_enable.mode)));
}
