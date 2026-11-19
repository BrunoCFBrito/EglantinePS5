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

TEST_CASE("A vertex attribute format write keeps the chain only with dynamic vertex input",
          "[video_core][fast_draw]") {
    // With dynamic vertex input the format is a draw parameter and TakeReusableDraw compares the
    // type class that stays in the key (Lichii #1519 / EglantinePS5 #129 series).
    const u32 format = MAXWELL3D_REG_INDEX(vertex_attrib_format);
    const bool previous = VideoCommon::Nx2FastDraw::g_dynamic_vertex_input.exchange(false);
    REQUIRE_FALSE(Tegra::Engines::Maxwell3D::IsFastDrawParameter(format + 8));
    VideoCommon::Nx2FastDraw::g_dynamic_vertex_input.store(true);
    REQUIRE(Tegra::Engines::Maxwell3D::IsFastDrawParameter(format + 8));
    REQUIRE(Tegra::Engines::Maxwell3D::IsFastDrawParameter(
        format + Tegra::Engines::Maxwell3D::Regs::NumVertexAttributes - 1));
    REQUIRE_FALSE(Tegra::Engines::Maxwell3D::IsFastDrawParameter(
        format + Tegra::Engines::Maxwell3D::Regs::NumVertexAttributes));
    VideoCommon::Nx2FastDraw::g_dynamic_vertex_input.store(previous);
}

TEST_CASE("A pipeline register_count write keeps the fast draw chain", "[video_core][fast_draw]") {
    // Written per draw, read by nothing (OoT: ~5,000/s of breaks on the console).
    const u32 pipelines = MAXWELL3D_REG_INDEX(pipelines);
    const u32 words = sizeof(Tegra::Engines::Maxwell3D::Regs::Pipeline) / sizeof(u32);
    REQUIRE(Tegra::Engines::Maxwell3D::IsFastDrawParameter(pipelines + 3));
    REQUIRE(Tegra::Engines::Maxwell3D::IsFastDrawParameter(pipelines + words + 3));
    REQUIRE_FALSE(Tegra::Engines::Maxwell3D::IsFastDrawParameter(pipelines + 1));
    REQUIRE_FALSE(Tegra::Engines::Maxwell3D::IsFastDrawParameter(
        pipelines + Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram * words));
}
