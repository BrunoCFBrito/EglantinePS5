// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <span>

#include <boost/container/small_vector.hpp>

#include "common/common_types.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/types.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

using Shader::Backend::SPIRV::NUM_TEXTURE_AND_IMAGE_SCALING_WORDS;

[[nodiscard]] inline std::optional<PixelFormat> PixelFormatFromImageFormat(
    Shader::ImageFormat format) {
    switch (format) {
    case Shader::ImageFormat::Typeless:
        return std::nullopt;
    case Shader::ImageFormat::R8_UINT:
        return PixelFormat::R8_UINT;
    case Shader::ImageFormat::R8_SINT:
        return PixelFormat::R8_SINT;
    case Shader::ImageFormat::R16_UINT:
        return PixelFormat::R16_UINT;
    case Shader::ImageFormat::R16_SINT:
        return PixelFormat::R16_SINT;
    case Shader::ImageFormat::R32_UINT:
        return PixelFormat::R32_UINT;
    case Shader::ImageFormat::R32G32_UINT:
        return PixelFormat::R32G32_UINT;
    case Shader::ImageFormat::R32G32B32A32_UINT:
        return PixelFormat::R32G32B32A32_UINT;
    }
    return std::nullopt;
}

[[nodiscard]] inline VkDeviceSize DescriptorSizeForType(const Device& device,
                                                        VkDescriptorType type) {
    const auto& props = device.DescriptorBufferProperties();
    const bool robust = device.IsRobustBufferAccessEnabled();
    switch (type) {
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        return robust ? props.robustUniformBufferDescriptorSize : props.uniformBufferDescriptorSize;
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        return robust ? props.robustStorageBufferDescriptorSize : props.storageBufferDescriptorSize;
    case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        return robust ? props.robustUniformTexelBufferDescriptorSize
                      : props.uniformTexelBufferDescriptorSize;
    case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        return robust ? props.robustStorageTexelBufferDescriptorSize
                      : props.storageTexelBufferDescriptorSize;
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        return props.combinedImageSamplerDescriptorSize;
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        return props.storageImageDescriptorSize;
    default:
        return 0;
    }
}

struct DescriptorBufferBinding {
    VkDescriptorType type;
    u32 count;
    VkDeviceSize offset;
    VkDeviceSize stride;
};

struct DescriptorBufferLayout {
    VkDeviceSize size{};
    boost::container::small_vector<DescriptorBufferBinding, 32> bindings;

    [[nodiscard]] bool Empty() const noexcept {
        return bindings.empty();
    }
};

inline void WriteDescriptorBuffer(const Device& device, const DescriptorBufferLayout& layout,
                                  const DescriptorUpdateEntry* payload, u8* host) {
    const vk::Device& dev = device.GetLogical();
    for (const DescriptorBufferBinding& binding : layout.bindings) {
        for (u32 index = 0; index < binding.count; ++index) {
            const DescriptorUpdateEntry& entry = *(payload++);
            const VkDescriptorAddressInfoEXT address_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
                .pNext = nullptr,
                .address = entry.address.address,
                .range = entry.address.range,
                .format = entry.address.format,
            };
            const VkDescriptorAddressInfoEXT* const buffer_info{
                entry.address.address != 0 ? &address_info : nullptr};
            VkDescriptorGetInfoEXT get_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
                .pNext = nullptr,
                .type = binding.type,
                .data{},
            };
            switch (binding.type) {
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                get_info.data.pUniformBuffer = buffer_info;
                break;
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                get_info.data.pStorageBuffer = buffer_info;
                break;
            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
                get_info.data.pUniformTexelBuffer = buffer_info;
                break;
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                get_info.data.pStorageTexelBuffer = buffer_info;
                break;
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                get_info.data.pCombinedImageSampler = &entry.image;
                break;
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                get_info.data.pStorageImage = &entry.image;
                break;
            default:
                continue;
            }
            dev.GetDescriptorEXT(get_info, binding.stride,
                                 host + binding.offset + index * binding.stride);
        }
    }
}

[[nodiscard]] inline u32 NumDescriptorEntries(const Shader::Info& info) {
    return Shader::NumDescriptors(info.constant_buffer_descriptors) +
           Shader::NumDescriptors(info.storage_buffers_descriptors) +
           Shader::NumDescriptors(info.texture_buffer_descriptors) +
           Shader::NumDescriptors(info.image_buffer_descriptors) +
           Shader::NumDescriptors(info.texture_descriptors) +
           Shader::NumDescriptors(info.image_descriptors);
}

class DescriptorLayoutBuilder {
public:
    DescriptorLayoutBuilder(const Device& device_) : device{&device_} {}

    bool CanUsePushDescriptor() const noexcept {
        if (!device->IsKhrPushDescriptorSupported() ||
            num_descriptors > device->MaxPushDescriptors()) {
            return false;
        }
        return !device->IsExtDescriptorBufferSupported() ||
               device->DescriptorBufferProperties().bufferlessPushDescriptors;
    }

    bool CanUseDescriptorBuffer() const noexcept {
        const auto& props = device->DescriptorBufferProperties();
        if (!device->IsExtDescriptorBufferSupported() || bindings.empty() ||
            !props.combinedImageSamplerDescriptorSingleArray) {
            return false;
        }
        return !props.bufferlessPushDescriptors || !CanUsePushDescriptor();
    }

    DescriptorBufferLayout MakeDescriptorBufferLayout(VkDescriptorSetLayout layout) const {
        DescriptorBufferLayout result;
        if (!layout) {
            return result;
        }
        const vk::Device& dev = device->GetLogical();
        result.size = dev.GetDescriptorSetLayoutSizeEXT(layout);
        result.bindings.reserve(bindings.size());
        for (const VkDescriptorSetLayoutBinding& entry : bindings) {
            result.bindings.push_back(DescriptorBufferBinding{
                .type = entry.descriptorType,
                .count = entry.descriptorCount,
                .offset = dev.GetDescriptorSetLayoutBindingOffsetEXT(layout, entry.binding),
                .stride = DescriptorSizeForType(*device, entry.descriptorType),
            });
        }
        return result;
    }

    vk::DescriptorSetLayout CreateDescriptorSetLayout(bool use_push_descriptor,
                                                      bool use_descriptor_buffer = false) const {
        if (bindings.empty()) {
            return nullptr;
        }
        VkDescriptorSetLayoutCreateFlags flags = 0;
        if (use_push_descriptor) {
            flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        }
        if (use_descriptor_buffer) {
            flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
        }
        boost::container::small_vector<VkDescriptorBindingFlags, 32> binding_flags;
        VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_ci{};
        const void* pnext = nullptr;
        if (!use_push_descriptor && device->IsDescriptorBindingPartiallyBoundSupported()) {
            binding_flags.assign(bindings.size(), VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
            binding_flags_ci = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
                .pNext = nullptr,
                .bindingCount = static_cast<u32>(binding_flags.size()),
                .pBindingFlags = binding_flags.data(),
            };
            pnext = &binding_flags_ci;
        }
        return device->GetLogical().CreateDescriptorSetLayout({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = pnext,
            .flags = flags,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        });
    }

    vk::DescriptorUpdateTemplate CreateTemplate(VkDescriptorSetLayout descriptor_set_layout,
                                                VkPipelineLayout pipeline_layout,
                                                bool use_push_descriptor) const {
        if (entries.empty()) {
            return nullptr;
        }
        const VkDescriptorUpdateTemplateType type =
            use_push_descriptor ? VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR
                                : VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;
        return device->GetLogical().CreateDescriptorUpdateTemplate({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .descriptorUpdateEntryCount = static_cast<u32>(entries.size()),
            .pDescriptorUpdateEntries = entries.data(),
            .templateType = type,
            .descriptorSetLayout = descriptor_set_layout,
            .pipelineBindPoint =
                is_compute ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS,
            .pipelineLayout = pipeline_layout,
            .set = 0,
        });
    }

    vk::PipelineLayout CreatePipelineLayout(VkDescriptorSetLayout descriptor_set_layout) const {
        using Shader::Backend::SPIRV::RenderAreaLayout;
        using Shader::Backend::SPIRV::RescalingLayout;
        const u32 size_offset = is_compute ? sizeof(RescalingLayout::down_factor) : 0u;
        const VkPushConstantRange range{
            .stageFlags = static_cast<VkShaderStageFlags>(
                is_compute ? VK_SHADER_STAGE_COMPUTE_BIT
                : mesh_emulation
                    ? VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT
                    : (uses_mesh ? VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_TASK_BIT_EXT |
                                       VK_SHADER_STAGE_MESH_BIT_EXT
                                 : VK_SHADER_STAGE_ALL_GRAPHICS)),
            .offset = 0,
            .size = static_cast<u32>(sizeof(RescalingLayout)) - size_offset +
                    static_cast<u32>(sizeof(RenderAreaLayout)),
        };
        return device->GetLogical().CreatePipelineLayout({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = descriptor_set_layout ? 1U : 0U,
            .pSetLayouts = bindings.empty() ? nullptr : &descriptor_set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
        });
    }

    /// Mesh emulation: a graphics pipeline layout whose mesh program runs as a compute shader
    /// (its bindings carry VK_SHADER_STAGE_COMPUTE_BIT and so does the push constant range).
    void EnableMeshEmulation() noexcept {
        mesh_emulation = true;
    }

    /// Mesh emulation: the output storage buffer the compute shader writes and the vertex
    /// shader reads, at an explicit binding (after every program's bindings).
    void AddMeshEmulationOutput(u32 output_binding) {
        bindings.push_back({
            .binding = output_binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        });
        entries.push_back({
            .dstBinding = output_binding,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .offset = offset,
            .stride = sizeof(DescriptorUpdateEntry),
        });
        binding = (std::max)(binding, output_binding + 1);
        ++num_descriptors;
        offset += sizeof(DescriptorUpdateEntry);
    }

    void Add(const Shader::Info& info, VkShaderStageFlags stage) {
        is_compute |= (stage & VK_SHADER_STAGE_COMPUTE_BIT) != 0 && !mesh_emulation;
        uses_mesh |= (stage & (VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT)) != 0;

        Add(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stage, info.constant_buffer_descriptors);
        Add(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage, info.storage_buffers_descriptors);
        Add(VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, stage, info.texture_buffer_descriptors);
        Add(VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, stage, info.image_buffer_descriptors);
        Add(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage, info.texture_descriptors);
        Add(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage, info.image_descriptors);
    }

private:
    template <typename Descriptors>
    void Add(VkDescriptorType type, VkShaderStageFlags stage, const Descriptors& descriptors) {
        const size_t num{descriptors.size()};
        for (size_t i = 0; i < num; ++i) {
            bindings.push_back({
                .binding = binding,
                .descriptorType = type,
                .descriptorCount = descriptors[i].count,
                .stageFlags = stage,
                .pImmutableSamplers = nullptr,
            });
            entries.push_back({
                .dstBinding = binding,
                .dstArrayElement = 0,
                .descriptorCount = descriptors[i].count,
                .descriptorType = type,
                .offset = offset,
                .stride = sizeof(DescriptorUpdateEntry),
            });
            ++binding;
            num_descriptors += descriptors[i].count;
            offset += sizeof(DescriptorUpdateEntry) * descriptors[i].count;
        }
    }

    const Device* device{};
    bool is_compute{};
    bool uses_mesh{};
    bool mesh_emulation{};
    boost::container::small_vector<VkDescriptorSetLayoutBinding, 32> bindings;
    boost::container::small_vector<VkDescriptorUpdateTemplateEntry, 32> entries;
    u32 binding{};
    u32 num_descriptors{};
    size_t offset{};
};

class RescalingPushConstant {
public:
    explicit RescalingPushConstant() noexcept {}

    void PushTexture(bool is_rescaled) noexcept {
        *texture_ptr |= is_rescaled ? texture_bit : 0u;
        texture_bit <<= 1u;
        if (texture_bit == 0u) {
            texture_bit = 1u;
            ++texture_ptr;
        }
    }

    void PushImage(bool is_rescaled) noexcept {
        *image_ptr |= is_rescaled ? image_bit : 0u;
        image_bit <<= 1u;
        if (image_bit == 0u) {
            image_bit = 1u;
            ++image_ptr;
        }
    }

    const std::array<u32, NUM_TEXTURE_AND_IMAGE_SCALING_WORDS>& Data() const noexcept {
        return words;
    }

    /// The words of an earlier draw's pushes (delta draws); push no more.
    void Assign(const std::array<u32, NUM_TEXTURE_AND_IMAGE_SCALING_WORDS>& data) noexcept {
        words = data;
    }

private:
    std::array<u32, NUM_TEXTURE_AND_IMAGE_SCALING_WORDS> words{};
    u32* texture_ptr{words.data()};
    u32* image_ptr{words.data() + Shader::Backend::SPIRV::NUM_TEXTURE_SCALING_WORDS};
    u32 texture_bit{1u};
    u32 image_bit{1u};
};

class RenderAreaPushConstant {
public:
    bool uses_render_area{};
    std::array<f32, 4> words{};
};

/// Reports a view id that names no live ImageView slot: a views entry the fill never wrote, or one
/// whose view was erased. Rate limited to the 1st, 2nd, 4th, 8th... report so a pipeline that keeps
/// producing broken views cannot flood the log. The EDEN_ prefix keeps the line in stderr.log,
/// which outlives the log sink's size cap.
inline void ReportBadImageView(bool compute, size_t stage, const char* kind, size_t descriptor,
                               u32 element, u32 count, VideoCommon::ImageViewId id,
                               size_t view_index, size_t view_count) noexcept {
    static std::atomic<u32> reported{};
    const u32 reports{reported.fetch_add(1, std::memory_order_relaxed) + 1};
    if ((reports & (reports - 1)) != 0) {
        return;
    }
    std::fprintf(stderr,
                 "EDEN_BAD_IMAGE_VIEW %s stage=%zu %s descriptor=%zu element=%u/%u view=%zu/%zu "
                 "id=0x%08x names no image view (report %u)\n",
                 compute ? "compute" : "graphics", stage, kind, descriptor, element, count,
                 view_index, view_count, static_cast<unsigned>(id.index), reports);
}

/// Checks the ids one stage's views hold. An id that names no live slot means the fill never wrote
/// that entry (or erased its view), and PushImageDescriptors would index slot_image_views with it:
/// release SlotVector::operator[] has no bounds check, so an unfilled entry (INVALID_INDEX) reads
/// values[0x7FFFFFFF] and faults, as in the 2026-11-16 crash. Reports the first bad entry and
/// returns false, so the caller skips the draw or dispatch instead of recording a wild binding.
/// view_index walks the stage's views in descriptor order across the four calls below.
[[nodiscard]] inline bool ValidateImageViews(
    const TextureCache& texture_cache, bool compute, size_t stage, const Shader::Info& info,
    std::span<const VideoCommon::ImageViewInOut> views, size_t& view_index) noexcept {
    const auto check{[&](const auto& descriptors, const char* kind) {
        for (size_t descriptor = 0; descriptor < descriptors.size(); ++descriptor) {
            const auto& desc{descriptors[descriptor]};
            for (u32 element = 0; element < desc.count; ++element, ++view_index) {
                if (view_index < views.size() &&
                    texture_cache.TryGetImageView(views[view_index].id) != nullptr) {
                    continue;
                }
                ReportBadImageView(compute, stage, kind, descriptor, element, desc.count,
                                   view_index < views.size() ? views[view_index].id
                                                             : VideoCommon::ImageViewId{},
                                   view_index, views.size());
                return false;
            }
        }
        return true;
    }};
    return check(info.texture_buffer_descriptors, "texture buffer") &&
           check(info.image_buffer_descriptors, "image buffer") &&
           check(info.texture_descriptors, "texture") &&
           check(info.image_descriptors, "image");
}

/// The views of a compute dispatch: every stage-less descriptor of its one shader info.
[[nodiscard]] inline bool ValidateImageViews(
    const TextureCache& texture_cache, const Shader::Info& info,
    std::span<const VideoCommon::ImageViewInOut> views) noexcept {
    size_t view_index{};
    return ValidateImageViews(texture_cache, true, 0, info, views, view_index);
}

/// The views of a graphics draw: stage_infos and enabled_stages are in shader-stage order and views
/// holds each enabled stage's texture buffers, image buffers, textures and images in that order
/// (the order ConfigureImpl's config_stage pushed them).
[[nodiscard]] inline bool ValidateImageViews(
    const TextureCache& texture_cache, std::span<const Shader::Info> stage_infos,
    std::span<const bool> enabled_stages,
    std::span<const VideoCommon::ImageViewInOut> views) noexcept {
    size_t view_index{};
    for (size_t stage = 0; stage < stage_infos.size(); ++stage) {
        if (!enabled_stages[stage]) {
            continue;
        }
        if (!ValidateImageViews(texture_cache, false, stage, stage_infos[stage], views,
                                view_index)) {
            return false;
        }
    }
    return true;
}

/// Whether a depth-compare sample of image_view is undefined and its handle garbage: its format
/// has no depth comparison on this device (what the sample returns then differs between vendors)
/// and it is an integer or block-compressed color texture, or its handle was read from memory (a
/// handle program). A single-channel color texture with a cbuf handle may be one a game compares
/// on purpose, and keeps its view.
[[nodiscard]] inline bool IsUncomparable(const ImageView& image_view, bool read_from_memory) {
    if (image_view.SupportsDepthComparison()) {
        return false;
    }
    const VideoCore::Surface::PixelFormat format{image_view.format};
    return read_from_memory || VideoCore::Surface::IsPixelFormatInteger(format) ||
           VideoCore::Surface::IsPixelFormatASTC(format) ||
           VideoCore::Surface::IsPixelFormatBCn(format) ||
           VideoCore::Surface::IsPixelFormatETC2(format);
}

inline void PushImageDescriptors(TextureCache& texture_cache,
                                 GuestDescriptorQueue& guest_descriptor_queue,
                                 const Shader::Info& info, RescalingPushConstant& rescaling,
                                 const VideoCommon::SamplerId*& samplers,
                                 const VideoCommon::ImageViewInOut*& views, size_t stage,
                                 bool compute) {
    const VideoCommon::ImageViewInOut* const views_begin{views};
    const size_t stage_views{Shader::NumDescriptors(info.texture_buffer_descriptors) +
                             Shader::NumDescriptors(info.image_buffer_descriptors) +
                             Shader::NumDescriptors(info.texture_descriptors) +
                             Shader::NumDescriptors(info.image_descriptors)};
    const u32 num_texture_buffers = Shader::NumDescriptors(info.texture_buffer_descriptors);
    const u32 num_image_buffers = Shader::NumDescriptors(info.image_buffer_descriptors);
    views += num_texture_buffers;
    views += num_image_buffers;
    size_t descriptor{};
    for (const auto& desc : info.texture_descriptors) {
        bool is_rescaled{};
        for (u32 index = 0; index < desc.count; ++index) {
            VideoCommon::ImageViewId image_view_id{(views++)->id};
            const VideoCommon::SamplerId sampler_id{*(samplers++)};
            ImageView* image_view{texture_cache.TryGetImageView(image_view_id)};
            if (image_view == nullptr) {
                // An unfilled views entry: its id names no slot. Bind the null view instead of
                // indexing slot_image_views with the wild id (the crash this guards).
                ReportBadImageView(compute, stage, "texture", descriptor, index, desc.count,
                                   image_view_id,
                                   static_cast<size_t>(views - views_begin) - 1, stage_views);
                image_view_id = VideoCommon::NULL_IMAGE_VIEW_ID;
                image_view = texture_cache.TryGetImageView(image_view_id);
            }
            if (image_view != nullptr && desc.is_depth &&
                image_view_id != VideoCommon::NULL_IMAGE_VIEW_ID &&
                IsUncomparable(*image_view, desc.handle_program != ~0U)) {
                image_view_id = VideoCommon::NULL_IMAGE_VIEW_ID;
                image_view = texture_cache.TryGetImageView(image_view_id);
            }
            if (image_view == nullptr) {
                // No null view either: emit a null descriptor rather than a wild lookup.
                guest_descriptor_queue.AddSampledImage(VK_NULL_HANDLE, VK_NULL_HANDLE);
                continue;
            }
            VkImageView vk_image_view{image_view->Handle(desc.type)};
            if (vk_image_view == VK_NULL_HANDLE) {
                if (ImageView* const null_view{
                        texture_cache.TryGetImageView(VideoCommon::NULL_IMAGE_VIEW_ID)}) {
                    vk_image_view = null_view->Handle(desc.type);
                }
            }
            Sampler& sampler{texture_cache.GetSampler(sampler_id)};
            guest_descriptor_queue.AddSampledImage(vk_image_view,
                                                   sampler.HandleFor(*image_view, desc.is_depth));
            const bool element_rescaled{texture_cache.IsRescaling(*image_view)};
            is_rescaled |= element_rescaled;
        }
        rescaling.PushTexture(is_rescaled);
        ++descriptor;
    }
    descriptor = 0;
    for (const auto& desc : info.image_descriptors) {
        bool is_rescaled{};
        for (u32 index = 0; index < desc.count; ++index) {
            const VideoCommon::ImageViewId image_view_id{(views++)->id};
            ImageView* image_view{texture_cache.TryGetImageView(image_view_id)};
            if (image_view == nullptr) {
                // Same guard as the sampled loop: a null storage image, and nothing was written so
                // there is no modification. StorageView is not asked for a view of a wild id.
                ReportBadImageView(compute, stage, "image", descriptor, index, desc.count,
                                   image_view_id,
                                   static_cast<size_t>(views - views_begin) - 1, stage_views);
                guest_descriptor_queue.AddImage(VK_NULL_HANDLE);
                continue;
            }
            const VkImageView vk_image_view{
                image_view->StorageView(desc.type, desc.format, desc.is_byte_addressed)};
            if (desc.is_written) {
                texture_cache.MarkModification(image_view->image_id);
            }
            guest_descriptor_queue.AddImage(vk_image_view);
            const bool element_rescaled{texture_cache.IsRescaling(*image_view)};
            is_rescaled |= element_rescaled;
        }
        rescaling.PushImage(is_rescaled);
        ++descriptor;
    }
}

} // namespace Vulkan
