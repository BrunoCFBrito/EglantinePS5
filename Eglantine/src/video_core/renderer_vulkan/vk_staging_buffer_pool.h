// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <climits>
#include <functional>
#include <optional>
#include <vector>

#include "common/common_types.h"

#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

struct StagingBufferRef {
    VkBuffer buffer;
    VkDeviceAddress device_address;
    VkDeviceSize offset;
    std::span<u8> mapped_span;
    MemoryUsage usage;
    u32 log2_level;
    u64 index;
};

/// What releasing kept staging buffers let go of.
struct StagingReleased {
    size_t buffers = 0;
    u64 bytes = 0;
    StagingReleased& operator+=(const StagingReleased& other) noexcept {
        buffers += other.buffers;
        bytes += other.bytes;
        return *this;
    }
};

class StagingBufferPool {
public:
    static constexpr size_t NUM_SYNCS = 16;

    explicit StagingBufferPool(const Device& device, MemoryAllocator& memory_allocator,
                               Scheduler& scheduler);
    ~StagingBufferPool();

    StagingBufferRef Request(size_t size, MemoryUsage usage, bool deferred = false);
    /// Request, or nothing when the device still refuses it after the pool's own recovery. A
    /// caller whose work can wait (a readback, a cache write-back) uses this instead of Request,
    /// which throws out of the GPU thread. `where` names it in EDEN_STAGING_SKIP.
    std::optional<StagingBufferRef> TryRequest(size_t size, MemoryUsage usage,
                                               bool deferred = false, const char* where = nullptr);

    void FreeDeferred(StagingBufferRef& ref);

    /// Keeps `ref` in the pool until the GPU is past `tick`. A transfer that reads the buffer in a
    /// later tick than the one it was requested in holds it this way: the ASTC transcode's decode
    /// runs after the pending-transcode waits have submitted the requesting tick.
    void KeepUntil(const StagingBufferRef& ref, u64 tick);

    [[nodiscard]] VkBuffer StreamBuf() const noexcept {
        return *stream_buffer;
    }

    void TickFrame();
    /// Lets go of every kept buffer the GPU is done with, and returns what went.
    StagingReleased ReleaseCompleted();

    /// Out of device memory: waits for the GPU, then frees every staging buffer it is done with
    /// except those handed out since the last submit. Returns what went.
    StagingReleased ReleaseMemory();

    /// Runs when memory is refused even after ReleaseMemory: frees what the caches hold and can
    /// let go (RasterizerVulkan sets it to the texture cache's ReclaimMemory). Returns the bytes
    /// it gave back; 0 when it could not run.
    std::function<u64()> reclaim;

    /// Bytes of the cached staging buffers of a kind, idle or in use (EDEN_NX2_MEM_TRACE).
    [[nodiscard]] u64 CachedBytes(MemoryUsage usage);

private:
    struct StreamBufferCommit {
        size_t upper_bound;
        u64 tick;
    };

    struct StagingBuffer {
        vk::Buffer buffer;
        VkDeviceAddress device_address;
        std::span<u8> mapped_span;
        MemoryUsage usage;
        u32 log2_level;
        u64 index;
        u64 tick = 0;
        bool deferred{};

        StagingBufferRef Ref() const noexcept {
            return {
                .buffer = *buffer,
                .device_address = device_address,
                .offset = 0,
                .mapped_span = mapped_span,
                .usage = usage,
                .log2_level = log2_level,
                .index = index,
            };
        }
    };

    struct StagingBuffers {
        std::vector<StagingBuffer> entries;
        size_t delete_index = 0;
        size_t iterate_index = 0;
    };

    static constexpr size_t NUM_LEVELS = sizeof(size_t) * CHAR_BIT;
    using StagingBuffersCache = std::array<StagingBuffers, NUM_LEVELS>;

    StagingBufferRef GetStreamBuffer(size_t size);

    bool AreRegionsActive(size_t region_begin, size_t region_end) const;

    StagingBufferRef GetStagingBuffer(size_t size, MemoryUsage usage, bool deferred = false);

    std::optional<StagingBufferRef> TryGetReservedBuffer(size_t size, MemoryUsage usage,
                                                         bool deferred);

    StagingBufferRef CreateStagingBuffer(size_t size, MemoryUsage usage, bool deferred);

    std::optional<StagingBufferRef> TryCreateStagingBuffer(size_t size, MemoryUsage usage,
                                                           bool deferred, const char* where);

    StagingBuffersCache& GetCache(MemoryUsage usage);

    void ReleaseCache(MemoryUsage usage);

    /// Frees every staging buffer the GPU is done with, of every kind (out of memory).
    StagingReleased ReleaseIdleBuffers();

    void ReleaseLevel(StagingBuffersCache& cache, size_t log2);

    /// Lets go of idle buffers, largest first, until the caches hold at most `cap` bytes.
    /// `why` names the caller in EDEN_STAGING_CAP, which logs the release.
    StagingReleased ReleaseIdleToCap(u64 cap, const char* why);

    size_t Region(size_t iter) const noexcept {
        return iter / region_size;
    }

    const Device& device;
    MemoryAllocator& memory_allocator;
    Scheduler& scheduler;

    vk::Buffer stream_buffer;
    VkDeviceAddress stream_buffer_address{};
    std::span<u8> stream_pointer;
    VkDeviceSize stream_buffer_size;
    VkDeviceSize region_size;

    size_t iterator = 0;
    size_t used_iterator = 0;
    size_t free_iterator = 0;
    std::array<u64, NUM_SYNCS> sync_ticks{};

    StagingBuffersCache device_local_cache;
    StagingBuffersCache upload_cache;
    StagingBuffersCache download_cache;

    size_t current_delete_level = 0;
    u64 buffer_index = 0;
    u64 unique_ids{};
};

} // namespace Vulkan
