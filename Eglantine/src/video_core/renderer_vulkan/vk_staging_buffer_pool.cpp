// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <utility>
#include <vector>

#include <fmt/ranges.h>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/bit_util.h"
#include "common/common_types.h"
#include "common/literals.h"
#include "common/logging.h"
#include "video_core/nx2_stutter_trace.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include <atomic>
#include <cstdio>
#include "performance.h"
#include "staging_buffers.h"

namespace Vulkan {
namespace {

using namespace Common::Literals;

// Maximum potential alignment of a Vulkan buffer
constexpr VkDeviceSize MAX_ALIGNMENT = 256;

// Stream buffer size in bytes
// *NIX drivers are more sensitive to increased buffers for streaming.
// Windows ones however, can intake bigger buffers and generally do not OOM.
// - GTX 960 on Windows will not OOM with 256mib
// - GT 1030 on ^NIX will OOM with 256mib
// The console's toolchain predefines __FreeBSD__ (9), so the PS5 took the FreeBSD value and
// kept 128 MiB bound for good in the direct-memory pool the CPU shares with the GPU (one VMA
// type 3 allocation of 128 MiB in every RADV_PS5_MEMORY sample) where 64 MiB were intended.
// EDEN_STREAM_BUFFER_MIB overrides; the 288 MiB shadow-map uploads go to dedicated staging.
VkDeviceSize MaxStreamBufferSize() {
    static const VkDeviceSize size = [] {
        if (const char* const text = std::getenv("EDEN_STREAM_BUFFER_MIB"); text != nullptr) {
            char* end = nullptr;
            const unsigned long long mib = std::strtoull(text, &end, 10);
            if (end != text && *end == '\0' && mib != 0 && mib <= 512) {
                return VkDeviceSize{mib} << 20;
            }
        }
#if defined(__FreeBSD__) && !defined(__PROSPERO__)
        return VkDeviceSize{128} << 20;
#else
        return VkDeviceSize{64} << 20;
#endif
    }();
    return size;
}

// Unused space after the end of the stream buffer. Constant buffers are bound from it and the
// shaders declare each one at its full 64 KiB, so Metal checks 64 KiB from the binding offset
// against the end of the buffer ("argument c3[0] ... has space for 26624 bytes, but argument has a
// length(65536)"). The tail is never handed out, so every binding has the full 64 KiB behind it.
constexpr VkDeviceSize STREAM_BUFFER_TAIL = 64_KiB;

size_t GetStreamBufferSize(const Device& device) {
    const VkDeviceSize max_size = MaxStreamBufferSize();
    if (!device.HasDebuggingToolAttached()) {
        if (device.IsIntegrated()) {
            // An integrated GPU's device-local host-visible heap can be no bigger than the stream
            // buffer (Radeon 780M: 256 MiB, 243 MiB budget), which then sat over that budget.
            // Leave a quarter of it free.
            VkDeviceSize heap_size = 0;
            ForEachDeviceLocalHostVisibleHeap(
                device, [&heap_size](size_t, VkMemoryHeap& heap) {
                    heap_size = (std::max)(heap_size, heap.size);
                });
            if (heap_size != 0 && heap_size <= max_size) {
                return Common::AlignUp(heap_size * 3 / 4, MAX_ALIGNMENT);
            }
        }
        return max_size;
    }

    VkDeviceSize size{0};
    bool has_device_local_host_visible_heap{};
    ForEachDeviceLocalHostVisibleHeap(device, [&size, &has_device_local_host_visible_heap](
                                                  size_t index, VkMemoryHeap& heap) {
        has_device_local_host_visible_heap = true;
        size = (std::max)(size, heap.size);
    });
    if (has_device_local_host_visible_heap) {
        // If rebar is not supported, cut the max heap size to 40%. This will allow 2 captures to be
        // loaded at the same time in RenderDoc. If rebar is supported, this shouldn't be an issue
        // as the heap will be much larger.
        if (size <= max_size) {
            size = size * 40 / 100;
        }
    } else {
        size = max_size;
    }
    return (std::min)(Common::AlignUp(size, MAX_ALIGNMENT), max_size);
}

// Byte cap over the three staging caches together. The pool a title gets on the console is small
// (Ocarina of Time: 12,224 MiB) and the caches held up to 264 MiB of device-local buffers and
// 176 MiB of uploads besides. EDEN_STAGING_CAP_MIB overrides the cap in MiB; 0 lifts it.
constexpr u64 DEFAULT_STAGING_CAP_BYTES = u64{192} << 20;
constexpr u64 NO_STAGING_CAP_BYTES = ~u64{0};

// An idle buffer larger than this is let go on the frame its tick completes, not on its level's
// next sweep (up to 64 frames later). One 12288x12288 D16 shadow map is 288 MiB.
constexpr u64 LARGE_IDLE_RELEASE_BYTES = u64{32} << 20;

// What the caches may keep idle while the pool is short. With nothing kept, every transfer made a
// buffer and let it go again, and each of those is a driver allocation that maps fresh memory: in
// a long Ocarina of Time session the pool stayed short for minutes and the back thread spent its
// frames there. EDEN_STAGING_SHORT_MIB overrides it in MiB (0 keeps nothing).
constexpr u64 DEFAULT_SHORT_STAGING_CAP_BYTES = u64{32} << 20;

u64 GetShortStagingCapBytes() {
    static const u64 cap = [] {
        const char* const text = std::getenv("EDEN_STAGING_SHORT_MIB");
        if (text == nullptr || text[0] == '\0') {
            return DEFAULT_SHORT_STAGING_CAP_BYTES;
        }
        char* end = nullptr;
        const unsigned long long mib = std::strtoull(text, &end, 10);
        if (end == text || *end != '\0' || mib > 1024) {
            return DEFAULT_SHORT_STAGING_CAP_BYTES;
        }
        return u64{mib} << 20;
    }();
    return cap;
}

u64 GetStagingCapBytes() {
    static const u64 cap = [] {
        const char* const text = std::getenv("EDEN_STAGING_CAP_MIB");
        if (text == nullptr || text[0] == '\0') {
            return DEFAULT_STAGING_CAP_BYTES;
        }
        char* end = nullptr;
        const unsigned long long mib = std::strtoull(text, &end, 10);
        if (end == text || *end != '\0') {
            return DEFAULT_STAGING_CAP_BYTES;
        }
        if (mib == 0 || mib > (NO_STAGING_CAP_BYTES >> 20)) {
            return NO_STAGING_CAP_BYTES;
        }
        return u64{mib} << 20;
    }();
    return cap;
}

// EDEN_STAGING_SKIP: a caller whose work can wait was refused a staging buffer even after the
// pool's recovery. The 1st, 2nd, 4th, 8th ... skip is logged.
void LogStagingSkip(const char* where, size_t size, MemoryUsage usage, bool deferred) {
    static std::atomic<unsigned long long> skips{0};
    const auto times = skips.fetch_add(1, std::memory_order_relaxed) + 1;
    if ((times & (times - 1)) == 0) {
        std::fprintf(stderr,
                     "EDEN_STAGING_SKIP where=%s asked=%zu usage=%u deferred=%d (%llu so far)\n",
                     where, size, static_cast<unsigned>(usage), int(deferred), times);
    }
}
} // Anonymous namespace

StagingBufferPool::StagingBufferPool(const Device& device_, MemoryAllocator& memory_allocator_,
                                     Scheduler& scheduler_)
    : device{device_}, memory_allocator{memory_allocator_}, scheduler{scheduler_},
      stream_buffer_size{GetStreamBufferSize(device)}, region_size{stream_buffer_size /
                                                                   StagingBufferPool::NUM_SYNCS} {
    VkBufferCreateInfo stream_ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = stream_buffer_size + STREAM_BUFFER_TAIL,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    if (device.IsExtTransformFeedbackSupported()) {
        stream_ci.usage |= VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT;
    }
    if (device.IsBufferDeviceAddressSupported()) {
        stream_ci.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    stream_buffer = memory_allocator.CreateBuffer(stream_ci, MemoryUsage::Stream);
    if (device.HasDebuggingToolAttached()) {
        stream_buffer.SetObjectNameEXT("Stream Buffer");
    }
    if (device.IsBufferDeviceAddressSupported()) {
        stream_buffer_address = device.GetLogical().GetBufferDeviceAddress(*stream_buffer);
    }
    stream_pointer = stream_buffer.Mapped();
    ASSERT_MSG(!stream_pointer.empty(), "Stream buffer must be host visible!");
}

StagingBufferPool::~StagingBufferPool() = default;

StagingBufferRef StagingBufferPool::Request(size_t size, MemoryUsage usage, bool deferred) {
    if (!deferred && usage == MemoryUsage::Upload && size <= region_size) {
        return GetStreamBuffer(size);
    }
    return GetStagingBuffer(size, usage, deferred);
}

std::optional<StagingBufferRef> StagingBufferPool::TryRequest(size_t size, MemoryUsage usage,
                                                              bool deferred, const char* where) {
    // No stream fast path here: a caller that can wait wants a dedicated buffer it can give
    // back, not a stream region it would have to keep for a tick.
    if (std::optional<StagingBufferRef> ref = TryGetReservedBuffer(size, usage, deferred)) {
        return ref;
    }
    return TryCreateStagingBuffer(size, usage, deferred, where);
}

void StagingBufferPool::FreeDeferred(StagingBufferRef& ref) {
    auto& entries = GetCache(ref.usage)[ref.log2_level].entries;
    const auto is_this_one = [&ref](const StagingBuffer& entry) {
        return entry.index == ref.index;
    };
    auto it = std::find_if(entries.begin(), entries.end(), is_this_one);
    ASSERT(it != entries.end());
    ASSERT(it->deferred);
    it->tick = scheduler.CurrentTick();
    it->deferred = false;
}

void StagingBufferPool::KeepUntil(const StagingBufferRef& ref, u64 tick) {
    if (ref.buffer == *stream_buffer) {
        // A stream buffer region is reused once the GPU is past the sync tick it was handed out
        // with: keep every region the transfer spans until `tick`.
        const size_t begin = (std::min)(static_cast<size_t>(ref.offset / region_size), NUM_SYNCS);
        const size_t end = (std::min)(
            static_cast<size_t>((ref.offset + ref.mapped_span.size_bytes() + region_size - 1) /
                                region_size),
            NUM_SYNCS);
        for (size_t i = begin; i < end; ++i) {
            sync_ticks[i] = (std::max)(sync_ticks[i], tick);
        }
        return;
    }
    auto& entries = GetCache(ref.usage)[ref.log2_level].entries;
    const auto is_this_one = [&ref](const StagingBuffer& entry) {
        return entry.index == ref.index;
    };
    const auto it = std::find_if(entries.begin(), entries.end(), is_this_one);
    ASSERT(it != entries.end());
    it->tick = (std::max)(it->tick, tick);
}

StagingReleased StagingBufferPool::ReleaseCompleted() {
    const auto idle = [this](const StagingBuffer& entry) {
        return !entry.deferred && scheduler.IsFree(entry.tick);
    };
    StagingReleased released;
    for (StagingBuffersCache* const cache : {&device_local_cache, &upload_cache, &download_cache}) {
        for (StagingBuffers& level : *cache) {
            const ::Eden::StagingBuffers::Released level_released =
                ::Eden::StagingBuffers::ReleaseFrom(level, idle);
            released.buffers += level_released.buffers;
            released.bytes += level_released.bytes;
        }
    }
    return released;
}

StagingReleased StagingBufferPool::ReleaseMemory() {
    // Waits for the submitted ticks only (#1997, Lichii). This runs while a draw or dispatch is
    // being set up, and Scheduler::Finish submits the open command buffer: the draw's index,
    // vertex and descriptor binds recorded so far would go into that submission. Buffers handed
    // out in the open tick stay busy anyway (their tick has not been submitted), so they are
    // neither freed nor handed out again.
    const u64 current_tick = scheduler.CurrentTick();
    if (current_tick > 0) {
        // Scheduler::Wait flushes when asked for a tick at or after the open one; ask only for
        // the last submitted tick.
        scheduler.Wait(current_tick - 1);
    }
    return ReleaseIdleBuffers();
}

void StagingBufferPool::TickFrame() {
    current_delete_level = (current_delete_level + 1) % NUM_LEVELS;

    ReleaseCache(MemoryUsage::DeviceLocal);
    ReleaseCache(MemoryUsage::Upload);
    ReleaseCache(MemoryUsage::Download);

    const auto idle = [this](const StagingBuffer& entry) {
        return !entry.deferred && scheduler.IsFree(entry.tick);
    };
    const auto bytes_of = [](const StagingBuffer& entry) {
        return ::Eden::StagingBuffers::Holds(entry.mapped_span.size(), entry.log2_level);
    };
    StagingBuffersCache* const caches[] = {&device_local_cache, &upload_cache, &download_cache};

    // Large idle buffers go on the frame their tick completes. Their level's sweep is up to 64
    // frames away, and a large level is usually one buffer.
    const auto is_large_idle = [&idle, &bytes_of](const StagingBuffer& entry) {
        return idle(entry) && bytes_of(entry) > LARGE_IDLE_RELEASE_BYTES;
    };
    for (StagingBuffersCache* const cache : caches) {
        for (StagingBuffers& level : *cache) {
            ::Eden::StagingBuffers::ReleaseFrom(level, is_large_idle);
        }
    }

    // Over the cap: let go of the largest idle buffers until the caches fit again. While the pool
    // is short only a small cap is kept (GetShortStagingCapBytes): the game's memory and the next
    // allocation need the rest more than the next transfer of the same size does.
    const bool short_memory = ::Eden::Performance::GraphicsMemoryIsShort();
    ReleaseIdleToCap(short_memory ? GetShortStagingCapBytes() : GetStagingCapBytes(),
                     short_memory ? "short" : "frame");
}

StagingReleased StagingBufferPool::ReleaseIdleToCap(u64 cap, const char* why) {
    const auto idle = [this](const StagingBuffer& entry) {
        return !entry.deferred && scheduler.IsFree(entry.tick);
    };
    const auto bytes_of = [](const StagingBuffer& entry) {
        return ::Eden::StagingBuffers::Holds(entry.mapped_span.size(), entry.log2_level);
    };
    StagingBuffersCache* const caches[] = {&device_local_cache, &upload_cache, &download_cache};

    // Buffers in use and deferred ones are counted, they hold memory, but they are never released
    // here: only idle buffers can go while the caches are over the cap.
    struct IdleBuffer {
        u64 index;
        u64 bytes;
    };
    std::vector<IdleBuffer> idle_buffers;
    u64 held = 0;
    for (StagingBuffersCache* const cache : caches) {
        for (const StagingBuffers& level : *cache) {
            for (const StagingBuffer& entry : level.entries) {
                const u64 bytes = bytes_of(entry);
                held += bytes;
                if (idle(entry)) {
                    idle_buffers.push_back({entry.index, bytes});
                }
            }
        }
    }
    if (held <= cap) {
        return {};
    }
    std::sort(idle_buffers.begin(), idle_buffers.end(),
              [](const IdleBuffer& a, const IdleBuffer& b) { return a.bytes > b.bytes; });
    u64 planned_bytes = 0;
    std::vector<u64> doomed;
    for (const IdleBuffer& candidate : idle_buffers) {
        if (held - planned_bytes <= cap) {
            break;
        }
        planned_bytes += candidate.bytes;
        doomed.push_back(candidate.index);
    }
    if (doomed.empty()) {
        return {};
    }
    std::sort(doomed.begin(), doomed.end());
    const auto drop = [this, &doomed](const StagingBuffer& entry) {
        return !entry.deferred && scheduler.IsFree(entry.tick) &&
               std::binary_search(doomed.begin(), doomed.end(), entry.index);
    };
    StagingReleased released;
    for (StagingBuffersCache* const cache : caches) {
        for (StagingBuffers& level : *cache) {
            const ::Eden::StagingBuffers::Released level_released =
                ::Eden::StagingBuffers::ReleaseFrom(level, drop);
            released.buffers += level_released.buffers;
            released.bytes += level_released.bytes;
        }
    }
    static std::atomic<unsigned long long> rows{0};
    const auto row = rows.fetch_add(1, std::memory_order_relaxed);
    if (row < 64 || (row & 255) == 0) {
        std::fprintf(stderr,
                     "EDEN_STAGING_CAP cap=%llu MiB held=%llu MiB released=%zu buffers (%llu MiB) "
                     "left=%llu MiB why=%s\n",
                     static_cast<unsigned long long>(cap >> 20),
                     static_cast<unsigned long long>(held >> 20), released.buffers,
                     static_cast<unsigned long long>(released.bytes >> 20),
                     static_cast<unsigned long long>((held - released.bytes) >> 20), why);
    }
    return released;
}

StagingBufferRef StagingBufferPool::GetStreamBuffer(size_t size) {
    if (AreRegionsActive(Region(free_iterator) + 1,
                         (std::min)(Region(iterator + size) + 1, NUM_SYNCS))) {
        // Avoid waiting for the previous usages to be free
        return GetStagingBuffer(size, MemoryUsage::Upload);
    }
    const u64 current_tick = scheduler.CurrentTick();
    std::fill(sync_ticks.begin() + Region(used_iterator), sync_ticks.begin() + Region(iterator),
              current_tick);
    used_iterator = iterator;

    if (iterator + size >= stream_buffer_size) {
        std::fill(sync_ticks.begin() + Region(used_iterator), sync_ticks.begin() + NUM_SYNCS,
                  current_tick);
        if (AreRegionsActive(0, Region(size) + 1)) {
            // Avoid waiting for the previous usages to be free. The position is kept until the
            // first regions are free: moved to the start here, it would make the guard above
            // treat the first region as acquired and hand it out while it is still in use.
            return GetStagingBuffer(size, MemoryUsage::Upload);
        }
        used_iterator = 0;
        iterator = 0;
        free_iterator = size;
    } else {
        free_iterator = (std::max)(free_iterator, iterator + size);
    }
    const size_t offset = iterator;
    iterator = Common::AlignUp(iterator + size, MAX_ALIGNMENT);
    return StagingBufferRef{
        .buffer = *stream_buffer,
        .device_address = stream_buffer_address,
        .offset = static_cast<VkDeviceSize>(offset),
        .mapped_span = stream_pointer.subspan(offset, size),
        .usage{},
        .log2_level{},
        .index{},
    };
}

bool StagingBufferPool::AreRegionsActive(size_t region_begin, size_t region_end) const {
    const u64 gpu_tick = scheduler.GetMasterSemaphore().KnownGpuTick();
    return std::any_of(sync_ticks.begin() + region_begin, sync_ticks.begin() + region_end,
                       [gpu_tick](u64 sync_tick) { return gpu_tick < sync_tick; });
};

StagingBufferRef StagingBufferPool::GetStagingBuffer(size_t size, MemoryUsage usage,
                                                     bool deferred) {
    if (const std::optional<StagingBufferRef> ref = TryGetReservedBuffer(size, usage, deferred)) {
        return *ref;
    }
    return CreateStagingBuffer(size, usage, deferred);
}

std::optional<StagingBufferRef> StagingBufferPool::TryGetReservedBuffer(size_t size,
                                                                        MemoryUsage usage,
                                                                        bool deferred) {
    StagingBuffers& cache_level = GetCache(usage)[Common::Log2Ceil(size)];

    const auto is_free = [this, size](const StagingBuffer& entry) {
        return !entry.deferred && scheduler.IsFree(entry.tick) &&
               ::Eden::StagingBuffers::Holds(entry.mapped_span.size(), entry.log2_level) >= size;
    };
    auto& entries = cache_level.entries;
    const auto hint_it = entries.begin() + cache_level.iterate_index;
    auto it = std::find_if(entries.begin() + cache_level.iterate_index, entries.end(), is_free);
    if (it == entries.end()) {
        it = std::find_if(entries.begin(), hint_it, is_free);
        if (it == hint_it) {
            return std::nullopt;
        }
    }
    cache_level.iterate_index = std::distance(entries.begin(), it) + 1;
    it->tick = deferred ? (std::numeric_limits<u64>::max)() : scheduler.CurrentTick();
    ASSERT(!it->deferred);
    it->deferred = deferred;
    return it->Ref();
}

StagingBufferRef StagingBufferPool::CreateStagingBuffer(size_t size, MemoryUsage usage,
                                                        bool deferred) {
    std::optional<StagingBufferRef> ref = TryCreateStagingBuffer(size, usage, deferred, nullptr);
    if (!ref) {
        LOG_CRITICAL(Render_Vulkan, "No memory for a {} byte staging buffer (usage {})", size,
                     static_cast<u32>(usage));
        throw vk::Exception(VK_ERROR_OUT_OF_DEVICE_MEMORY);
    }
    return *ref;
}

std::optional<StagingBufferRef> StagingBufferPool::TryCreateStagingBuffer(size_t size,
                                                                          MemoryUsage usage,
                                                                          bool deferred,
                                                                          const char* where) {
    if (static unsigned rows{}; size >= (64ULL << 20) && rows++ < 96) {
        for (unsigned kind = 0; kind < 3; ++kind) {
            auto& cache = kind == 0 ? device_local_cache : kind == 1 ? upload_cache : download_cache;
            size_t count = 0, pinned = 0, busy = 0, free = 0;
            unsigned long long bytes = 0;
            for (const auto& level : cache) for (const auto& entry : level.entries) {
                ++count;
                bytes += ::Eden::StagingBuffers::Holds(entry.mapped_span.size(), entry.log2_level);
                if (entry.deferred) ++pinned;
                else if (scheduler.IsFree(entry.tick)) ++free;
                else ++busy;
            }
            std::fprintf(stderr, "MKW_STAGING_MISS asked=%zu usage=%u deferred=%d kind=%u count=%zu bytes=%llu pinned=%zu busy=%zu free=%zu tick=%llu completed=%llu\n",
                         size, unsigned(usage), int(deferred), kind, count, bytes, pinned, busy, free,
                         static_cast<unsigned long long>(scheduler.CurrentTick()),
                         static_cast<unsigned long long>(scheduler.GetMasterSemaphore().KnownGpuTick()));
        }
    }

    const VideoCommon::StutterTrace::SiteScope site{VideoCommon::StutterTrace::Site::StagingCreate};
    auto const log2_size = Common::Log2Ceil<u32>(u32(size));
    const bool exact = ::Eden::StagingBuffers::Exact(size, usage != MemoryUsage::DeviceLocal);
    const VkDeviceSize made = ::Eden::StagingBuffers::Size(size, log2_size, exact);
    if (exact) {
        // Kept buffers of this power of two too small for the transfer go before it is made.
        ::Eden::StagingBuffers::ReleaseFrom(GetCache(usage)[log2_size], [this, size](const StagingBuffer& entry) {
            return !entry.deferred && scheduler.IsFree(entry.tick) &&
                   ::Eden::StagingBuffers::Holds(entry.mapped_span.size(), entry.log2_level) < size;
        });
    }
    // The cap is enforced here too, not only in TickFrame: a burst inside one frame asked for
    // more than the cap at once, and buffers that became idle after the frame's own check were
    // still kept when the next allocation came. Under a short pool the smaller cap is enforced on
    // every request; otherwise the scan only runs for buffers of at least a MiB, the ones that can
    // take the pool, so small allocations stay as cheap as they were.
    {
        const bool short_memory = ::Eden::Performance::GraphicsMemoryIsShort();
        const u64 cap = short_memory ? GetShortStagingCapBytes() : GetStagingCapBytes();
        if (short_memory || made >= (VkDeviceSize{1} << 20)) {
            ReleaseIdleToCap(cap > made ? cap - made : 0, short_memory ? "short" : "request");
        }
    }
    VkBufferCreateInfo buffer_ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = made,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                 VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, // mesh emulation dispatch and draw arguments
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    if (device.IsExtTransformFeedbackSupported()) {
        buffer_ci.usage |= VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT;
    }
    if (device.IsBufferDeviceAddressSupported()) {
        buffer_ci.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    vk::Buffer buffer = memory_allocator.TryCreateBuffer(buffer_ci, usage);
    StagingReleased released;
    if (!buffer) {
        // No memory left for it (#262: a 128 MiB upload on an 8 GB laptop with Intel UHD). Free
        // the idle staging buffers and try again, then let ReleaseMemory wait for the submitted
        // ticks and free the ones the GPU was still reading too. Buffers handed out in the open
        // tick may feed commands not recorded yet and are left busy, so they are neither freed
        // nor handed out again.
        released += ReleaseCompleted();
        buffer = memory_allocator.TryCreateBuffer(buffer_ci, usage);
        if (!buffer) {
            released += ReleaseMemory();
            buffer = memory_allocator.TryCreateBuffer(buffer_ci, usage);
        }
        // The caches hold more than the staging pool: the texture cache keeps images it has
        // already deleted alive for the GPU, and the collector's own readback buffers are
        // deferred. Ask it for that memory once, then try once more.
        if (!buffer && reclaim) {
            const u64 reclaimed = reclaim();
            if (reclaimed != 0) {
                static std::atomic<unsigned long long> rows{0};
                const auto row = rows.fetch_add(1, std::memory_order_relaxed);
                if (row < 32 || (row & 255) == 0) {
                    std::fprintf(stderr,
                                 "EDEN_STAGING_RECLAIM reclaimed=%llu MiB asked=%zu usage=%u\n",
                                 static_cast<unsigned long long>(reclaimed >> 20), size,
                                 static_cast<unsigned>(usage));
                }
                buffer = memory_allocator.TryCreateBuffer(buffer_ci, usage);
            }
        }
    }
    if (!buffer) {
        std::fprintf(stderr, "EDEN_STAGING_OOM asked=%zu made=%llu released=%zu buffers (%llu MiB)\n",
                     size, static_cast<unsigned long long>(made), released.buffers,
                     static_cast<unsigned long long>(released.bytes >> 20));
        static u32 logged{};
        if (logged++ == 0) {
            LogMemoryTrace(device);
        }
        // The submitted ticks are done after ReleaseMemory, so the stream buffer after its last
        // range is free. The ranges the open tick handed out may hold data for commands not
        // recorded yet, so it does not wrap. A deferred upload cannot use it (FreeDeferred looks
        // the buffer up in the upload cache), so only a non-deferred upload is served from the
        // tail.
        if (usage == MemoryUsage::Upload && !deferred && iterator + size < stream_buffer_size) {
            if (logged <= 32) {
                LOG_ERROR(Render_Vulkan,
                          "No memory for a {} byte upload buffer; using the stream buffer", size);
            }
            const size_t offset = iterator;
            free_iterator = (std::max)(free_iterator, offset + size);
            iterator = Common::AlignUp(offset + size, MAX_ALIGNMENT);
            return StagingBufferRef{
                .buffer = *stream_buffer,
                .device_address = stream_buffer_address,
                .offset = static_cast<VkDeviceSize>(offset),
                .mapped_span = stream_pointer.subspan(offset, size),
                .usage{},
                .log2_level{},
                .index{},
            };
        }
        // A caller whose transfer can wait gets nothing back instead of an exception out of the
        // GPU thread; Request still throws (the wrapper above).
        if (where != nullptr) {
            LogStagingSkip(where, size, usage, deferred);
        }
        return std::nullopt;
    }
    // A large buffer smaller than the power of two: the 1st, 2nd, 4th, 8th ... in the log.
    if (exact && made >= (VkDeviceSize{64} << 20) && made < (VkDeviceSize{1} << log2_size)) {
        static std::atomic<unsigned long long> large{0};
        const auto times = large.fetch_add(1, std::memory_order_relaxed) + 1;
        if ((times & (times - 1)) == 0) {
            std::fprintf(stderr, "EDEN_STAGING_EXACT %llu MiB made for a transfer of %llu MiB, not the %llu MiB of the "
                         "next power of two (%llu so far)\n",
                         static_cast<unsigned long long>(made >> 20), static_cast<unsigned long long>(size >> 20),
                         static_cast<unsigned long long>((VkDeviceSize{1} << log2_size) >> 20), times);
        }
    }
    if (device.HasDebuggingToolAttached()) {
        ++buffer_index;
        buffer.SetObjectNameEXT(fmt::format("Staging Buffer {}", buffer_index).c_str());
    }
    const std::span<u8> mapped_span = buffer.Mapped();
    const VkDeviceAddress buffer_address =
        device.IsBufferDeviceAddressSupported()
            ? device.GetLogical().GetBufferDeviceAddress(*buffer)
            : VkDeviceAddress{};
    StagingBuffer& entry = GetCache(usage)[log2_size].entries.emplace_back(StagingBuffer{
        .buffer = std::move(buffer),
        .device_address = buffer_address,
        .mapped_span = mapped_span,
        .usage = usage,
        .log2_level = log2_size,
        .index = unique_ids++,
        .tick = deferred ? (std::numeric_limits<u64>::max)() : scheduler.CurrentTick(),
        .deferred = deferred,
    });
    return entry.Ref();
}

StagingBufferPool::StagingBuffersCache& StagingBufferPool::GetCache(MemoryUsage usage) {
    switch (usage) {
    case MemoryUsage::DeviceLocal:
        return device_local_cache;
    case MemoryUsage::Upload:
        return upload_cache;
    case MemoryUsage::Download:
        return download_cache;
    default:
        ASSERT_MSG(false, "Invalid memory usage={}", usage);
        return upload_cache;
    }
}

u64 StagingBufferPool::CachedBytes(MemoryUsage usage) {
    u64 bytes{};
    const StagingBuffersCache& cache = GetCache(usage);
    for (size_t level = 0; level < NUM_LEVELS; ++level) {
        if (!cache[level].entries.empty()) {
            bytes += u64{cache[level].entries.size()} << level;
        }
    }
    return bytes;
}

StagingReleased StagingBufferPool::ReleaseIdleBuffers() {
    const auto idle = [this](const StagingBuffer& entry) {
        return !entry.deferred && scheduler.IsFree(entry.tick);
    };
    StagingReleased released;
    for (StagingBuffersCache* const cache : {&device_local_cache, &upload_cache, &download_cache}) {
        for (StagingBuffers& staging : *cache) {
            const ::Eden::StagingBuffers::Released level_released =
                ::Eden::StagingBuffers::ReleaseFrom(staging, idle);
            released.buffers += level_released.buffers;
            released.bytes += level_released.bytes;
            staging.delete_index = 0;
            staging.iterate_index = 0;
        }
    }
    return released;
}

void StagingBufferPool::ReleaseCache(MemoryUsage usage) {
    ReleaseLevel(GetCache(usage), current_delete_level);
}

void StagingBufferPool::ReleaseLevel(StagingBuffersCache& cache, size_t log2) {
    constexpr size_t deletions_per_tick = 16;
    auto& staging = cache[log2];
    auto& entries = staging.entries;
    const size_t old_size = entries.size();

    // Keep two idle upload buffers per level up to 32 MiB (under 128 MiB in all). Each level is
    // swept every 64 frames, so after a pause each upload over 8 MiB (past the stream buffer's
    // regions) got a new dedicated allocation, written to for the first time (16 MiB on an
    // RTX 4080, idle system: 0.4 ms to create, 1.2 ms more for the first write).
    constexpr size_t KeepMaxLog2 = 25;
    constexpr size_t KeepPerLevel = 2;
    if (&cache == &upload_cache && log2 <= KeepMaxLog2) {
        size_t idle = 0;
        std::erase_if(entries, [this, &idle](const StagingBuffer& entry) {
            return scheduler.IsFree(entry.tick) && ++idle > KeepPerLevel;
        });
        staging.delete_index = 0;
        if (staging.iterate_index > entries.size()) {
            staging.iterate_index = 0;
        }
        return;
    }

    const auto is_deletable = [this](const StagingBuffer& entry) {
        return scheduler.IsFree(entry.tick);
    };
    const size_t begin_offset = staging.delete_index;
    const size_t end_offset = (std::min)(begin_offset + deletions_per_tick, old_size);
    const auto begin = entries.begin() + begin_offset;
    const auto end = entries.begin() + end_offset;
    entries.erase(std::remove_if(begin, end, is_deletable), end);

    const size_t new_size = entries.size();
    staging.delete_index += deletions_per_tick;
    if (staging.delete_index >= new_size) {
        staging.delete_index = 0;
    }
    if (staging.iterate_index > new_size) {
        staging.iterate_index = 0;
    }
}

} // namespace Vulkan
