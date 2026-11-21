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

TEST_CASE("Query, condition address and unused register writes keep the fast draw chain",
          "[video_core][fast_draw]") {
    // OoT writes the report semaphore (query, address, payload), render_enable.address_low and
    // reads-nothing registers thousands of times a second; none is in what a delta draw keeps.
    // Measured on the console: state-changed full draws fell 22,953 -> 18,513/s at the spawn.
    using Regs = Tegra::Engines::Maxwell3D::Regs;
    const auto check = [](u32 method) {
        return Tegra::Engines::Maxwell3D::IsFastDrawParameter(method);
    };
    REQUIRE(check(MAXWELL3D_REG_INDEX(report_semaphore.query)));
    REQUIRE(check(MAXWELL3D_REG_INDEX(report_semaphore.address_low)));
    REQUIRE(check(MAXWELL3D_REG_INDEX(report_semaphore.payload)));
    REQUIRE(check(MAXWELL3D_REG_INDEX(render_enable.address_high)));
    REQUIRE(check(MAXWELL3D_REG_INDEX(render_enable.address_low)));
    REQUIRE(check(MAXWELL3D_REG_INDEX(const_color_rendering)));
    // Pipeline words 7.. (reserved[2..3] and the padding); word 6 is the program address low.
    const u32 pipelines = MAXWELL3D_REG_INDEX(pipelines);
    const u32 words = sizeof(Regs::Pipeline) / sizeof(u32);
    REQUIRE(check(pipelines + 7));
    REQUIRE(check(pipelines + words + words - 1));
    REQUIRE_FALSE(check(pipelines + 6));
    // The transform feedback gap: buffers end, controls begin 0x300 bytes later.
    const u32 tf = MAXWELL3D_REG_INDEX(transform_feedback);
    const u32 tf_buffers = sizeof(Regs::TransformFeedback::Buffer) / sizeof(u32);
    REQUIRE(check(tf + tf_buffers * Regs::NumTransformFeedbackBuffers));
    REQUIRE(check(tf + tf_buffers * Regs::NumTransformFeedbackBuffers + 0x100 / sizeof(u32) - 1));
    REQUIRE_FALSE(check(tf)); // buffer enable: read (Address/IsEnabled)
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
