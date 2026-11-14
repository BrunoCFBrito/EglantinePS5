// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Tegra::Texture::ASTC {

/// `buffer_row_blocks` is the input row pitch in blocks (0: the extent's own width). Rows of
/// an unswizzled run are padded to whole GOB blocks, so it can be larger than the extent.
void Decompress(std::span<const uint8_t> data, uint32_t width, uint32_t height, uint32_t depth,
                uint32_t block_width, uint32_t block_height, std::span<uint8_t> output,
                uint32_t buffer_row_blocks = 0);

/// True if no 16-byte block in `data` decodes to alpha below 1.
bool IsOpaque(std::span<const uint8_t> data);

} // namespace Tegra::Texture::ASTC
