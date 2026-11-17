// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <array>
#include <bit>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <dxgi1_4.h>
#endif

#include <cstring>
#include <iterator>
#include <string>

#include <fmt/format.h>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/common_types.h"
#include "common/literals.h"
#include "common/logging.h"
#include "common/memory_detect.h"
#include <ranges>
#include "video_core/vulkan_common/unbacked_fallback.h"
#include "video_core/vulkan_common/vma.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "performance.h"
namespace Common { void GuestMemoryUsage(std::size_t* span, std::size_t* held) noexcept; }
extern "C" std::size_t eden_heap_committed(void);
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "common/settings.h"

// The guest memory service (PS5 port) gives memory back on demand: it trims the driver's unused
// blocks, then packs guest memory the game has not written for a while. Absent elsewhere. Returns
// the bytes it gave back. (Old core patch 0023, lost in the move to the in-repo core.)
extern "C" [[gnu::weak]] std::size_t eden_memory_recover(std::size_t bytes) noexcept;

namespace Vulkan {
namespace {
// Out of device memory: asks for memory back (`bytes` and a 64 MiB block more) and tries again, up
// to three times. The pool the console shares with the guest's memory fills with cold guest
// memory that can be packed; without this the first failure skipped an image or ended the app.
// Upstream: heydemoura/EglantinePS5 PR #113, commit bd24db9.
template <typename Create>
VkResult RecoverDeviceMemory(VkResult result, VkDeviceSize bytes, Create&& create) {
    for (int attempt = 0; result == VK_ERROR_OUT_OF_DEVICE_MEMORY && attempt < 3; ++attempt) {
        if (&eden_memory_recover == nullptr ||
            eden_memory_recover(static_cast<std::size_t>(bytes) + (std::size_t{64} << 20)) == 0) {
            break;
        }
        result = create();
    }
    return result;
}

// Helpers translating MemoryUsage to flags/usage

    [[maybe_unused]] VkMemoryPropertyFlags MemoryUsagePropertyFlags(MemoryUsage usage) {
        switch (usage) {
            case MemoryUsage::DeviceLocal:
                return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            case MemoryUsage::Upload:
                return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            case MemoryUsage::Download:
                return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                        VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
            case MemoryUsage::Stream:
                return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        }
        ASSERT_MSG(false, "Invalid memory usage={}", usage);
        return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }

    [[nodiscard]] VkMemoryPropertyFlags MemoryUsagePreferredVmaFlags(MemoryUsage usage) {
        if (usage == MemoryUsage::Download) {
            return VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        }
        return usage != MemoryUsage::DeviceLocal ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                                    : VkMemoryPropertyFlagBits{};
    }

    [[nodiscard]] VmaAllocationCreateFlags MemoryUsageVmaFlags(MemoryUsage usage) {
        switch (usage) {
            case MemoryUsage::Upload:
            case MemoryUsage::Stream:
                return VMA_ALLOCATION_CREATE_MAPPED_BIT |
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            case MemoryUsage::Download:
                return VMA_ALLOCATION_CREATE_MAPPED_BIT |
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
            case MemoryUsage::DeviceLocal:
                return {};
        }
        return {};
    }

    [[nodiscard]] VmaMemoryUsage MemoryUsageVma(MemoryUsage usage) {
        switch (usage) {
            case MemoryUsage::DeviceLocal:
            case MemoryUsage::Stream:
                return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            case MemoryUsage::Upload:
            case MemoryUsage::Download:
                return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        }
        return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    }
} // namespace

//MemoryCommit is now VMA-backed
MemoryCommit::MemoryCommit(VmaAllocator alloc, VmaAllocation a,
                            const VmaAllocationInfo &info) noexcept
        : allocator{alloc}, allocation{a}, memory{info.deviceMemory},
            offset{info.offset}, size{info.size}, mapped_ptr{info.pMappedData} {
    // Log GPU memory allocation
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_memory_tracking.GetValue()) {
        GPU::Logging::GPULogger::GetInstance().LogMemoryAllocation(
            reinterpret_cast<uintptr_t>(memory),
            static_cast<u64>(size),
            0  // Memory property flags (not easily available from VMA)
        );
    }
}

MemoryCommit::~MemoryCommit() { Release(); }

MemoryCommit::MemoryCommit(MemoryCommit &&rhs) noexcept
        : allocator{std::exchange(rhs.allocator, nullptr)},
            allocation{std::exchange(rhs.allocation, nullptr)},
            memory{std::exchange(rhs.memory, VK_NULL_HANDLE)},
            offset{std::exchange(rhs.offset, 0)},
            size{std::exchange(rhs.size, 0)},
            mapped_ptr{std::exchange(rhs.mapped_ptr, nullptr)} {}

MemoryCommit &MemoryCommit::operator=(MemoryCommit &&rhs) noexcept {
    if (this != &rhs) {
        Release();
        allocator = std::exchange(rhs.allocator, nullptr);
        allocation = std::exchange(rhs.allocation, nullptr);
        memory = std::exchange(rhs.memory, VK_NULL_HANDLE);
        offset = std::exchange(rhs.offset, 0);
        size = std::exchange(rhs.size, 0);
        mapped_ptr = std::exchange(rhs.mapped_ptr, nullptr);
    }
    return *this;
}

std::span<u8> MemoryCommit::Map()
{
    if (!allocation) return {};
    if (!mapped_ptr) {
        if (vmaMapMemory(allocator, allocation, &mapped_ptr) != VK_SUCCESS) return {};
    }
    const size_t n = static_cast<size_t>(std::min<VkDeviceSize>(size,
                                                                (std::numeric_limits<size_t>::max)()));
    return std::span<u8>{static_cast<u8 *>(mapped_ptr), n};
}

std::span<const u8> MemoryCommit::Map() const
{
    if (!allocation) return {};
    if (!mapped_ptr) {
        void *p = nullptr;
        if (vmaMapMemory(allocator, allocation, &p) != VK_SUCCESS) return {};
        const_cast<MemoryCommit *>(this)->mapped_ptr = p;
    }
    const size_t n = static_cast<size_t>(std::min<VkDeviceSize>(size,
                                                                (std::numeric_limits<size_t>::max)()));
    return std::span<const u8>{static_cast<const u8 *>(mapped_ptr), n};
}

void MemoryCommit::Unmap()
{
    if (allocation && mapped_ptr) {
        vmaUnmapMemory(allocator, allocation);
        mapped_ptr = nullptr;
    }
}

void MemoryCommit::Release() {
    if (allocation && allocator) {
        // Log GPU memory deallocation
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_memory_tracking.GetValue() &&
            memory != VK_NULL_HANDLE) {
            GPU::Logging::GPULogger::GetInstance().LogMemoryDeallocation(
                reinterpret_cast<uintptr_t>(memory)
            );
        }

        if (mapped_ptr) {
            vmaUnmapMemory(allocator, allocation);
            mapped_ptr = nullptr;
        }
        vmaFreeMemory(allocator, allocation);
    }
    allocation = nullptr;
    allocator = nullptr;
    memory = VK_NULL_HANDLE;
    offset = 0;
    size = 0;
}

namespace {
// Image recycling (EDEN_NX2_IMAGE_RECYCLE_MIB, default 256, 0: off; EDEN_NX2_IMAGE_RECYCLE_MS,
// default 1000). Games render into transient targets at a different address every frame (Ocarina:
// a 960x540 RGBA16F colour target and a 256x256 depth one, up to 45 a second, 2026-11-16), so the
// texture cache deletes last frame's images as superseded and makes new ones. Each new one took
// fresh device memory and freeing it left the blocks 0.4-1.1 GB empty. A freed image with memory
// of its own (VMA dedicates one when it is larger than half a block: 4 MiB with the 8 MiB blocks
// the console uses) of 4 MiB and more is kept instead of destroyed and handed back as it is to
// the next request with an identical create info. A new image's contents are undefined and every
// first use transitions it from VK_IMAGE_LAYOUT_UNDEFINED, so a recycled one behaves the same.
// Kept for EDEN_NX2_IMAGE_RECYCLE_MS at most and EDEN_NX2_IMAGE_RECYCLE_MIB in all, and never
// while the pool is critical: the kept images go back to the driver then (the game's memory is
// worth more than a reuse), and nothing new is kept. Ported from lethevimlet/lichii-bc250
// 65c6b7790f, with the critical rule from bxymartin/PokeDinner b9e00405.
struct ImageRecyclePool {
    struct Key {
        std::vector<u32> words;
        bool operator==(const Key&) const = default;
    };
    struct Entry {
        Key key;
        VkImage image{};
        VmaAllocation allocation{};
        u64 size{};
        s64 since{}; ///< steady clock, nanoseconds, when the image was kept
    };
    std::mutex mutex;
    std::atomic<u64> cap{};   ///< bytes the kept images may hold in all (0: recycling off)
    u64 held{};               ///< bytes the kept images hold now
    VmaAllocator allocator{}; ///< the allocator the kept images belong to
    std::unordered_map<VkImage, std::pair<Key, u64>> candidates; ///< Recyclable images in use
    std::deque<Entry> free;                                      ///< Nobody uses them, oldest first
    u64 hits{}, kept{}, evicted{};
    std::chrono::steady_clock::time_point last_report{};
};
ImageRecyclePool g_image_pool;
constexpr u64 MinRecycledImageBytes = 4ull << 20;

s64 RecycleKeepNs() {
    static const s64 ns = [] {
        const char* value = std::getenv("EDEN_NX2_IMAGE_RECYCLE_MS");
        const unsigned long long ms =
            value != nullptr ? std::strtoull(value, nullptr, 10) : 1000ull;
        return static_cast<s64>(ms) * 1'000'000;
    }();
    return ns;
}

s64 RecycleNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// The create info as words, or nullopt when it carries anything but a view format list. The
/// initial layout must be UNDEFINED: a recycled image was left in whatever layout its last owner
/// put it in, and only then may its contents be discarded and used as a new image's are.
std::optional<ImageRecyclePool::Key> RecycleKey(const VkImageCreateInfo& ci) {
    if (ci.tiling != VK_IMAGE_TILING_OPTIMAL ||
        (ci.flags & (VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT |
                     VK_IMAGE_CREATE_SPARSE_ALIASED_BIT)) != 0 ||
        ci.sharingMode != VK_SHARING_MODE_EXCLUSIVE || ci.queueFamilyIndexCount != 0 ||
        ci.initialLayout != VK_IMAGE_LAYOUT_UNDEFINED) {
        return std::nullopt;
    }
    ImageRecyclePool::Key key;
    key.words = {static_cast<u32>(ci.flags),        static_cast<u32>(ci.imageType),
                 static_cast<u32>(ci.format),       ci.extent.width,
                 ci.extent.height,                  ci.extent.depth,
                 ci.mipLevels,                      ci.arrayLayers,
                 static_cast<u32>(ci.samples),      ci.usage,
                 static_cast<u32>(ci.initialLayout)};
    for (auto* next = static_cast<const VkBaseInStructure*>(ci.pNext); next != nullptr;
         next = next->pNext) {
        if (next->sType != VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO) {
            return std::nullopt;
        }
        const auto* list = reinterpret_cast<const VkImageFormatListCreateInfo*>(next);
        key.words.push_back(0xF0F0F0F0u);
        for (u32 i = 0; i < list->viewFormatCount; ++i) {
            key.words.push_back(static_cast<u32>(list->pViewFormats[i]));
        }
    }
    return key;
}

void DestroyRecycled(VmaAllocator allocator, std::vector<ImageRecyclePool::Entry>& entries) {
    for (const auto& entry : entries) {
        vmaDestroyImage(allocator, entry.image, entry.allocation);
    }
    entries.clear();
}

/// pool.mutex held: every kept image goes into `destroy`, which the caller destroys outside the
/// lock. Returns the bytes they held.
u64 MoveAllRecycledLocked(std::vector<ImageRecyclePool::Entry>& destroy) {
    auto& pool = g_image_pool;
    u64 bytes = 0;
    while (!pool.free.empty()) {
        bytes += pool.free.front().size;
        destroy.push_back(std::move(pool.free.front()));
        pool.free.pop_front();
    }
    pool.held = 0;
    return bytes;
}

/// pool.mutex held: the kept images past their time and the oldest above the cap go into
/// `destroy`; the bin is ordered by when images were kept, so both are at its front.
void ExpireRecycledLocked(std::vector<ImageRecyclePool::Entry>& destroy) {
    auto& pool = g_image_pool;
    const s64 now = RecycleNowNs();
    const s64 keep = RecycleKeepNs();
    const u64 cap = pool.cap.load(std::memory_order_relaxed);
    while (!pool.free.empty() &&
           (pool.held > cap || now - pool.free.front().since > keep)) {
        pool.held -= pool.free.front().size;
        destroy.push_back(std::move(pool.free.front()));
        pool.free.pop_front();
        ++pool.evicted;
    }
}

/// pool.mutex held: one line per 30 s on what the recycler keeps and has done.
void ReportRecycleLocked() {
    auto& pool = g_image_pool;
    const auto now = std::chrono::steady_clock::now();
    if (now - pool.last_report < std::chrono::seconds{30}) {
        return;
    }
    pool.last_report = now;
    LOG_INFO(Render_Vulkan, "Image recycling: {} reused, {} kept, {} gone after their time, {} MiB held",
             pool.hits, pool.kept, pool.evicted, pool.held >> 20);
}

/// A kept image with that create info, the one kept longest ago first. The caller gives it back
/// to a new owner: it is in use again, so a later release keeps it again.
bool TakeRecycled(const ImageRecyclePool::Key& key, VkImage& image, VmaAllocation& allocation) {
    auto& pool = g_image_pool;
    if (pool.cap.load(std::memory_order_relaxed) == 0) {
        return false;
    }
    std::vector<ImageRecyclePool::Entry> destroy;
    ImageRecyclePool::Entry found;
    VmaAllocator pool_allocator{};
    bool hit = false;
    {
        std::scoped_lock lock{pool.mutex};
        pool_allocator = pool.allocator;
        ExpireRecycledLocked(destroy);
        for (auto it = pool.free.begin(); it != pool.free.end(); ++it) {
            if (!(it->key == key)) {
                continue;
            }
            found = std::move(*it);
            pool.free.erase(it);
            pool.held -= found.size;
            pool.candidates.emplace(found.image, std::make_pair(found.key, found.size));
            ++pool.hits;
            hit = true;
            break;
        }
        ReportRecycleLocked();
    }
    if (pool_allocator != nullptr) {
        DestroyRecycled(pool_allocator, destroy);
    }
    if (hit) {
        image = found.image;
        allocation = found.allocation;
    }
    return hit;
}

/// vk::Image::Release's hook (vk::g_image_recycler): an image about to be destroyed is kept for
/// reuse when it is one of the candidates made with its own memory, and destroyed as usual when
/// false is returned. Called after the texture cache's destruction delay, so the GPU is done with
/// it. While the pool is critical nothing is kept and the images kept until now go back.
bool RecycleImage(VmaAllocator allocator, VkImage image, VmaAllocation allocation) noexcept {
    auto& pool = g_image_pool;
    std::vector<ImageRecyclePool::Entry> destroy;
    VmaAllocator pool_allocator{};
    bool kept = false;
    {
        std::scoped_lock lock{pool.mutex};
        if (pool.cap.load(std::memory_order_relaxed) == 0 || allocator != pool.allocator) {
            return false;
        }
        const auto it = pool.candidates.find(image);
        if (it == pool.candidates.end()) {
            return false;
        }
        ImageRecyclePool::Key key = std::move(it->second.first);
        const u64 size = it->second.second;
        pool.candidates.erase(it);
        pool_allocator = pool.allocator;
        if (::Eden::Performance::GraphicsMemoryIsCritical()) {
            // A recycler must never hold memory while the pool is critical: what it keeps goes
            // back to the driver, and this image is destroyed as usual by its owner.
            MoveAllRecycledLocked(destroy);
        } else {
            pool.free.push_back({std::move(key), image, allocation, size, RecycleNowNs()});
            pool.held += size;
            ++pool.kept;
            ExpireRecycledLocked(destroy);
            kept = true;
        }
        ReportRecycleLocked();
    }
    if (pool_allocator != nullptr) {
        DestroyRecycled(pool_allocator, destroy);
    }
    return kept;
}
} // Anonymous namespace

u64 FlushImageRecyclePool() {
    auto& pool = g_image_pool;
    std::vector<ImageRecyclePool::Entry> destroy;
    VmaAllocator pool_allocator{};
    u64 bytes = 0;
    {
        std::scoped_lock lock{pool.mutex};
        pool_allocator = pool.allocator;
        bytes = MoveAllRecycledLocked(destroy);
        if (bytes != 0) {
            ReportRecycleLocked();
        }
    }
    if (pool_allocator != nullptr) {
        DestroyRecycled(pool_allocator, destroy);
    }
    return bytes;
}

void ReleaseImageRecyclePool() {
    auto& pool = g_image_pool;
    std::vector<ImageRecyclePool::Entry> destroy;
    VmaAllocator pool_allocator{};
    {
        std::scoped_lock lock{pool.mutex};
        vk::g_image_recycler.store(nullptr, std::memory_order_release);
        pool_allocator = pool.allocator;
        MoveAllRecycledLocked(destroy);
        // The candidates are images their owners still hold: with the hook off, their own
        // destructors destroy them (the allocator pointer is cleared last).
        pool.candidates.clear();
        pool.cap.store(0, std::memory_order_relaxed);
        pool.allocator = nullptr;
    }
    if (pool_allocator != nullptr) {
        DestroyRecycled(pool_allocator, destroy);
    }
}

namespace {
std::mutex vma_probe_mutex;
VmaAllocator vma_probe_allocator{};
void EdenVmaSnapshot(VmaAllocator allocator, unsigned tag) {
    char* json = nullptr;
    vmaBuildStatsString(allocator, &json, VK_TRUE);
    if (!json) return;
    char path[192];
    std::snprintf(path, sizeof(path), "/data/eglantineps5/logs/mkw-vma-%d-%u.json", getpid(), tag);
    const auto bytes = std::strlen(json);
    bool written = false;
    if (bytes <= (8U << 20)) {
        if (FILE* file = std::fopen(path, "w")) {
            written = std::fwrite(json, 1, bytes, file) == bytes;
            std::fclose(file);
        }
    }
    std::fprintf(stderr, "MKW_VMA_MAP tag=%u bytes=%zu written=%u\n", tag, bytes, unsigned(written));
    vmaFreeStatsString(allocator, json);
}
}
extern "C" void eden_vma_probe(unsigned seconds) {
    std::lock_guard lock{vma_probe_mutex};
    if (vma_probe_allocator) EdenVmaSnapshot(vma_probe_allocator, 100000 + seconds);
}
MemoryAllocator::MemoryAllocator(const Device &device_)
        : device{device_}, allocator{device.GetAllocator()},
            properties{device_.GetPhysical().GetMemoryProperties().memoryProperties},
            buffer_image_granularity{
                    device_.GetPhysical().GetProperties().limits.bufferImageGranularity} {

    { std::lock_guard lock{vma_probe_mutex}; vma_probe_allocator = allocator; }
    // Preserve the previous "RenderDoc small heap" trimming behavior that we had in original vma minus the heap bug
    if (device.HasDebuggingToolAttached())
    {
        using namespace Common::Literals;
        ForEachDeviceLocalHostVisibleHeap(device, [this](size_t heap_idx, VkMemoryHeap &heap) {
            if (heap.size <= 256_MiB) {
                for (u32 t = 0; t < properties.memoryTypeCount; ++t) {
                    if (properties.memoryTypes[t].heapIndex == heap_idx) {
                        valid_memory_types &= ~(1u << t);
                    }
                }
            }
        });
    }
    {
        // Image recycling: hand freed images with their own memory back to the next request of
        // the same shape instead of destroying and remaking them (vulkan_wrapper.h's
        // g_image_recycler, RecycleImage above). EDEN_NX2_IMAGE_RECYCLE_MIB=0 turns it off.
        static const u64 recycle_cap = [] {
            const char* value = std::getenv("EDEN_NX2_IMAGE_RECYCLE_MIB");
            const unsigned long long mib =
                value != nullptr ? std::strtoull(value, nullptr, 10) : 256ull;
            return static_cast<u64>(mib) << 20;
        }();
        std::scoped_lock lock{g_image_pool.mutex};
        if (recycle_cap != 0 && g_image_pool.allocator == nullptr) {
            g_image_pool.cap.store(recycle_cap, std::memory_order_relaxed);
            g_image_pool.allocator = allocator;
            vk::g_image_recycler.store(RecycleImage, std::memory_order_release);
            LOG_INFO(Render_Vulkan,
                     "Image recycling: images of {} MiB and more with their own memory are kept up "
                     "to {} MiB and {} ms for reuse (EDEN_NX2_IMAGE_RECYCLE_MIB=0 turns it off)",
                     MinRecycledImageBytes >> 20, recycle_cap >> 20,
                     static_cast<unsigned long long>(RecycleKeepNs() / 1'000'000));
        }
    }
}

MemoryAllocator::~MemoryAllocator() {
    std::lock_guard lock{vma_probe_mutex};
    if (vma_probe_allocator == allocator) vma_probe_allocator = nullptr;
    for (auto pool : temporary_pools) if (pool) vmaDestroyPool(allocator, pool);
}

vk::Image MemoryAllocator::CreateImage(const VkImageCreateInfo &ci, bool dedicated) const
{
    vk::Image image;
    vk::Check(AllocateImage(ci, dedicated, false, image));
    return image;
}

vk::Image MemoryAllocator::TryCreateImage(const VkImageCreateInfo &ci, bool dedicated) const {
    vk::Image image;
    const VkResult result = AllocateImage(ci, dedicated, true, image);
    if (result != VK_ERROR_OUT_OF_DEVICE_MEMORY && result != VK_ERROR_OUT_OF_HOST_MEMORY) {
        vk::Check(result);
    }
    return image;
}

VkResult MemoryAllocator::AllocateImage(const VkImageCreateInfo &ci, bool dedicated, bool may_fail,
                                        vk::Image &image) const {
    const VmaAllocationCreateInfo alloc_ci = {
            // Best fit: images freed by the thousand leave holes in the blocks, which new ones
            // should fill before VMA takes another block.
            .flags = static_cast<VmaAllocationCreateFlags>(VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT |
                                                           VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT) |
                     (dedicated ? VmaAllocationCreateFlags{VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT}
                               : VmaAllocationCreateFlags{0}),
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .requiredFlags = 0,
            .preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            .memoryTypeBits = 0,
            .pool = VK_NULL_HANDLE,
            .pUserData = nullptr,
            .priority = 0.f,
    };

    VkImage handle{};
    VmaAllocation allocation{};
    VmaAllocationInfo alloc_info{};
    const u64 texels = u64{ci.extent.width} * ci.extent.height * ci.extent.depth * ci.arrayLayers;
    const bool huge = texels > MaxBackedImageTexels;
    // A target of the same shape kept from a moment ago is handed back as it is (RecycleImage).
    std::optional<ImageRecyclePool::Key> recycle_key;
    if (g_image_pool.cap.load(std::memory_order_relaxed) != 0 && !huge) {
        recycle_key = RecycleKey(ci);
        if (recycle_key && TakeRecycled(*recycle_key, handle, allocation)) {
            image = vk::Image(handle, ci.usage, *device.GetLogical(), allocator, allocation,
                              device.GetDispatchLoader());
            return VK_SUCCESS;
        }
    }
    bool host_placed = false;
    VkResult result = huge ? VK_ERROR_OUT_OF_DEVICE_MEMORY
                           : vmaCreateImage(allocator, &ci, &alloc_ci, &handle, &allocation, &alloc_info);
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY && !huge) {
        // The images the recycler keeps are memory their owners have already given up: give it
        // back to the driver before accepting that the pool is out, and try once more.
        if (const u64 released = FlushImageRecyclePool(); released != 0) {
            static std::atomic<u32> recycled_logged{0};
            if (recycled_logged.fetch_add(1, std::memory_order_relaxed) < 32) {
                LOG_INFO(Render_Vulkan,
                         "Image recycling: gave {} MiB back for a refused image, trying again",
                         released >> 20);
            }
            result = vmaCreateImage(allocator, &ci, &alloc_ci, &handle, &allocation, &alloc_info);
        }
    }
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY && !huge) {
        // Over the device memory budget (Switch 2 titles fill 8 GB cards): let the driver page
        // it out rather than failing the draw and aborting.
        static u32 logged{};
        if (logged++ < 32) {
            LOG_WARNING(Render_Vulkan,
                        "Image {}x{}x{} ({} levels, {} layers, format {}) is over the memory "
                        "budget; allocating it anyway",
                        ci.extent.width, ci.extent.height, ci.extent.depth, ci.mipLevels,
                        ci.arrayLayers, static_cast<u32>(ci.format));
        }
        pressure_events.fetch_add(1, std::memory_order_relaxed);
        // A driver allocation was refused (see graphics_memory_refusals): the pipeline cache's
        // on-demand eviction runs when it sees one.
        ::Eden::Performance::graphics_memory_refusals.fetch_add(1, std::memory_order_relaxed);
        VmaAllocationCreateInfo relaxed = alloc_ci;
        relaxed.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        result = vmaCreateImage(allocator, &ci, &relaxed, &handle, &allocation, &alloc_info);
        // The request is not known before a failure: 8 bytes a texel covers every format used.
        result = RecoverDeviceMemory(result, texels * 8, [&] {
            return vmaCreateImage(allocator, &ci, &relaxed, &handle, &allocation, &alloc_info);
        });
    }
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY && !huge) {
        // Still out of device memory: a real image in host memory (slow, but valid) beats one
        // with nothing bound, which AMD's drivers do not survive being sampled.
        VmaAllocationCreateInfo host = alloc_ci;
        host.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        host.preferredFlags = 0;
        result = vmaCreateImage(allocator, &ci, &host, &handle, &allocation, &alloc_info);
        static u32 host_logged{};
        if (result == VK_SUCCESS) {
            host_placed = true;
            if (host_logged++ < 16) {
                LOG_WARNING(Render_Vulkan,
                            "Image {}x{}x{} ({} levels, {} layers, format {}) is out of device "
                            "memory; placed in host memory",
                            ci.extent.width, ci.extent.height, ci.extent.depth, ci.mipLevels,
                            ci.arrayLayers, static_cast<u32>(ci.format));
            }
        }
    }
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY && !huge) {
        pressure_events.fetch_add(1, std::memory_order_relaxed);
    }
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY && ci.imageType != VK_IMAGE_TYPE_1D &&
        UsesUnbackedFallback(texels, may_fail)) {
        // Too big to back at all, e.g. title 1's sparse virtual texture arrays (8192x8192,
        // 68 layers, ~23 GB): on the console only resident tiles have memory. Use an image with
        // nothing bound, so views and copies still work, instead of aborting. A smaller image
        // gets one only for a caller that would throw otherwise.
        if (vk::Image sparse = CreateUnbackedImage(ci)) {
            LOG_WARNING(Render_Vulkan,
                        "Image {}x{}x{} ({} levels, {} layers, format {}) does not fit; using an "
                        "unbacked sparse image",
                        ci.extent.width, ci.extent.height, ci.extent.depth, ci.mipLevels,
                        ci.arrayLayers, static_cast<u32>(ci.format));
            image = std::move(sparse);
            return VK_SUCCESS;
        }
    }
    if (result != VK_SUCCESS) {
        failed_images.fetch_add(1, std::memory_order_relaxed);
        ::Eden::Performance::graphics_memory_refusals.fetch_add(1, std::memory_order_relaxed);
        static u32 failed_logged{};
        if (failed_logged == 0) {
            // What ran out: the device's heaps or the host's commit. Before the error line,
            // which is the one that reaches the log file when the session ends here.
            LogMemoryTrace(device);
        }
        if (failed_logged++ < 32) {
            LOG_ERROR(Render_Vulkan, "Image {}x{}x{} ({} levels, {} layers, format {}) failed: {}",
                      ci.extent.width, ci.extent.height, ci.extent.depth, ci.mipLevels,
                      ci.arrayLayers, static_cast<u32>(ci.format), static_cast<int>(result));
        }
        return result;
    }

    // Log GPU memory allocation for images
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_memory_tracking.GetValue()) {
        GPU::Logging::GPULogger::GetInstance().LogMemoryAllocation(
            reinterpret_cast<uintptr_t>(alloc_info.deviceMemory),
            static_cast<u64>(alloc_info.size),
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
    }

    if (recycle_key && allocation != nullptr && !host_placed &&
        alloc_info.size >= MinRecycledImageBytes) {
        VmaAllocationInfo2 info2{};
        vmaGetAllocationInfo2(allocator, allocation, &info2);
        // Its own memory: dedicated, or the only allocation in a block of exactly its size (VMA
        // makes one for allocations over the block size, and it goes back whole when freed).
        // Keeping such an image holds no more than its bytes.
        if (info2.dedicatedMemory || info2.blockSize == info2.allocationInfo.size) {
            std::scoped_lock lock{g_image_pool.mutex};
            g_image_pool.candidates.emplace(
                handle, std::make_pair(std::move(*recycle_key), alloc_info.size));
        }
    }
    image = vk::Image(handle, ci.usage, *device.GetLogical(), allocator, allocation,
                      device.GetDispatchLoader());
    return VK_SUCCESS;
}

vk::Image MemoryAllocator::CreateUnbackedImage(const VkImageCreateInfo &ci, bool aliased) const {
    if (ci.imageType == VK_IMAGE_TYPE_1D) {
        return vk::Image{};
    }
    VkImageCreateInfo sparse_ci = ci;
    sparse_ci.flags |= VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
    if (aliased) {
        sparse_ci.flags |= VK_IMAGE_CREATE_SPARSE_ALIASED_BIT;
    }
    VkImage sparse{};
    VkResult sparse_result = device.GetDispatchLoader().vkCreateImage(*device.GetLogical(), &sparse_ci, nullptr, &sparse);
    if (sparse_result != VK_SUCCESS) {
        VkImageCreateInfo binding_only = ci;
        binding_only.flags |= VK_IMAGE_CREATE_SPARSE_BINDING_BIT;
        const VkResult second = device.GetDispatchLoader().vkCreateImage(*device.GetLogical(), &binding_only, nullptr, &sparse);
        std::fprintf(stderr, "EDEN_SPARSE_IMAGE %ux%ux%u, %u layers, %u levels, format %d, usage 0x%x, flags 0x%x, tiling %d: "
                     "with residency %d, binding alone %d\n",
                     ci.extent.width, ci.extent.height, ci.extent.depth, ci.arrayLayers, ci.mipLevels, int(ci.format),
                     unsigned(ci.usage), unsigned(ci.flags), int(ci.tiling), int(sparse_result), int(second));
        if (second != VK_SUCCESS) {
            // Once: which smaller sizes does the driver accept? Each probe is made and destroyed at once.
            static std::atomic<bool> probed{false};
            if (!probed.exchange(true)) {
                for (u32 layers = ci.arrayLayers / 2; layers >= 1; layers /= 2) {
                    VkImageCreateInfo probe = sparse_ci;
                    probe.arrayLayers = layers;
                    VkImage probe_image{};
                    const VkResult probed_result =
                        device.GetDispatchLoader().vkCreateImage(*device.GetLogical(), &probe, nullptr, &probe_image);
                    std::fprintf(stderr, "EDEN_SPARSE_PROBE %u layers: %d\n", layers, int(probed_result));
                    if (probed_result == VK_SUCCESS) {
                        device.GetDispatchLoader().vkDestroyImage(*device.GetLogical(), probe_image, nullptr);
                    }
                    if (layers == 1) break;
                }
            }
            return vk::Image{};
        }
    }
    return vk::Image(sparse, ci.usage, *device.GetLogical(), allocator, VK_NULL_HANDLE,
                     device.GetDispatchLoader());
}

VmaAllocation MemoryAllocator::AllocateSparseMemory(const VkMemoryRequirements &requirements,
                                                   VkDeviceMemory &memory,
                                                   VkDeviceSize &offset) const {
    const VmaAllocationCreateInfo alloc_ci{
        .flags = 0,
        .usage = VMA_MEMORY_USAGE_UNKNOWN,
        .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        .preferredFlags = 0,
        .memoryTypeBits = requirements.memoryTypeBits,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
        .priority = 0.f,
    };
    VmaAllocation allocation{};
    VmaAllocationInfo info{};
    // In a pool of their own (see Device::SeparatePool); anything over half a block, or a pool
    // allocation that fails, takes the default blocks.
    u32 memory_type{};
    if (requirements.size <= Device::SeparatePoolBlockSize(Device::SeparatePool::SparsePages) / 2 &&
        vmaFindMemoryTypeIndex(allocator, requirements.memoryTypeBits, &alloc_ci, &memory_type) ==
            VK_SUCCESS) {
        if (const VmaPool pool =
                device.GetSeparatePool(Device::SeparatePool::SparsePages, memory_type)) {
            VmaAllocationCreateInfo pool_ci = alloc_ci;
            pool_ci.pool = pool;
            if (vmaAllocateMemory(allocator, &requirements, &pool_ci, &allocation, &info) ==
                VK_SUCCESS) {
                memory = info.deviceMemory;
                offset = info.offset;
                return allocation;
            }
        }
    }
    if (vmaAllocateMemory(allocator, &requirements, &alloc_ci, &allocation, &info) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    memory = info.deviceMemory;
    offset = info.offset;
    return allocation;
}

void MemoryAllocator::FreeSparseMemory(VmaAllocation allocation) const {
    if (allocation != VK_NULL_HANDLE) {
        vmaFreeMemory(allocator, allocation);
    }
}

bool MemoryAllocator::CreateBufferInSeparatePool(const VkBufferCreateInfo& ci, MemoryUsage usage,
                                                 const VmaAllocationCreateInfo& alloc_ci,
                                                 VkBuffer& handle, VmaAllocation& allocation,
                                                 VmaAllocationInfo& alloc_info) const {
    // The buffer cache's buffers, churned by its collector: in a pool of their own (see
    // Device::SeparatePool). Anything over half a block takes the default blocks (dedicated).
    if (usage != MemoryUsage::DeviceLocal ||
        ci.size > Device::SeparatePoolBlockSize(Device::SeparatePool::DeviceBuffers) / 2) {
        return false;
    }
    u32 memory_type{};
    {
        std::scoped_lock lk{buffer_type_mutex};
        const auto key = (u64{ci.usage} << 32) | ci.flags;
        if (const auto it = buffer_memory_types.find(key); it != buffer_memory_types.end()) {
            memory_type = it->second;
        } else if (vmaFindMemoryTypeIndexForBufferInfo(allocator, &ci, &alloc_ci, &memory_type) ==
                   VK_SUCCESS) {
            buffer_memory_types.emplace(key, memory_type);
        } else {
            return false;
        }
    }
    const VmaPool pool = device.GetSeparatePool(Device::SeparatePool::DeviceBuffers, memory_type);
    if (pool == VK_NULL_HANDLE) {
        return false;
    }
    VmaAllocationCreateInfo pool_ci = alloc_ci;
    pool_ci.pool = pool;
    pool_ci.memoryTypeBits = 0;
    return vmaCreateBuffer(allocator, &ci, &pool_ci, &handle, &allocation, &alloc_info) ==
           VK_SUCCESS;
}

vk::Buffer MemoryAllocator::CreateBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage) const {
    vk::Buffer buffer;
    vk::Check(AllocateBuffer(ci, usage, buffer));
    return buffer;
}

vk::Buffer MemoryAllocator::TryCreateBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage) const {
    vk::Buffer buffer;
    const VkResult result = AllocateBuffer(ci, usage, buffer);
    if (result != VK_ERROR_OUT_OF_DEVICE_MEMORY && result != VK_ERROR_OUT_OF_HOST_MEMORY) {
        vk::Check(result);
    }
    return buffer;
}

VkResult MemoryAllocator::AllocateBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage,
                                         vk::Buffer &buffer) const {
    // MESA will do memcpy() if not marked as host cached, so just force mark it for most buffers
    auto const anv_flags = (usage == MemoryUsage::Stream
        && device.GetDriverID() == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA)
        ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0;
    const VmaAllocationCreateInfo alloc_ci = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | MemoryUsageVmaFlags(usage),
        .usage = MemoryUsageVma(usage),
        .requiredFlags = 0,
        .preferredFlags = MemoryUsagePreferredVmaFlags(usage) | anv_flags,
        .memoryTypeBits = usage == MemoryUsage::Stream ? 0u : valid_memory_types,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
        .priority = 0.f,
    };

    VkBuffer handle{};
    VmaAllocationInfo alloc_info{};
    VmaAllocation allocation{};
    VkMemoryPropertyFlags property_flags{};

    if (!CreateBufferInSeparatePool(ci, usage, alloc_ci, handle, allocation, alloc_info)) {
        VkResult result =
            vmaCreateBuffer(allocator, &ci, &alloc_ci, &handle, &allocation, &alloc_info);
        if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY) {
            // Over the device memory budget, as images are in CreateImage: buffers have no
            // unbacked fallback, and the vk::Exception a failed staging or buffer cache
            // allocation throws from the GPU thread aborts the process. Phones share RAM with
            // the GPU, so the budget is often lower than the memory that is really free.
            static u32 logged{};
            if (logged++ < 32) {
                LOG_WARNING(Render_Vulkan,
                            "Buffer of {} bytes (usage {}) is over the memory budget; "
                            "allocating it anyway",
                            ci.size, static_cast<u32>(usage));
            }
            // The texture cache answers each pressure event by keeping more device memory free.
            // Upload and download buffers live in host memory on a discrete GPU, so they only
            // count when the heaps are shared or the buffer asked for device-local memory.
            // The pipeline cache's on-demand eviction answers the refusal itself, whatever the
            // usage (see graphics_memory_refusals).
            ::Eden::Performance::graphics_memory_refusals.fetch_add(1, std::memory_order_relaxed);
            if (device.IsIntegrated() || usage == MemoryUsage::DeviceLocal ||
                usage == MemoryUsage::Stream) {
                pressure_events.fetch_add(1, std::memory_order_relaxed);
            }
            VmaAllocationCreateInfo relaxed = alloc_ci;
            relaxed.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
            result = vmaCreateBuffer(allocator, &ci, &relaxed, &handle, &allocation, &alloc_info);
            result = RecoverDeviceMemory(result, ci.size, [&] {
                return vmaCreateBuffer(allocator, &ci, &relaxed, &handle, &allocation, &alloc_info);
            });
        }
        if (result != VK_SUCCESS) {
            ::Eden::Performance::graphics_memory_refusals.fetch_add(1, std::memory_order_relaxed);
            return result;
        }
    }
    vmaGetAllocationMemoryProperties(allocator, allocation, &property_flags);

    // Log GPU memory allocation for buffers
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_memory_tracking.GetValue()) {
        GPU::Logging::GPULogger::GetInstance().LogMemoryAllocation(
            reinterpret_cast<uintptr_t>(alloc_info.deviceMemory),
            static_cast<u64>(alloc_info.size),
            property_flags
        );
    }

    u8 *data = reinterpret_cast<u8 *>(alloc_info.pMappedData);
    const std::span<u8> mapped_data = data ? std::span<u8>{data, ci.size} : std::span<u8>{};
    const bool is_coherent = (property_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

    const vk::MemoryLocation location{
        .memory = alloc_info.deviceMemory,
        .offset = alloc_info.offset,
        .memory_type = alloc_info.memoryType,
    };
    buffer = vk::Buffer(handle, *device.GetLogical(), allocator, allocation, mapped_data,
                        is_coherent, location, device.GetDispatchLoader());
    return VK_SUCCESS;
}

vk::Buffer MemoryAllocator::CreateBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage,
                                         VkDeviceSize min_alignment) const {
    if (min_alignment <= 1) {
        return CreateBuffer(ci, usage);
    }
    VkMemoryPropertyFlags anv_flags = 0;
    if (usage == MemoryUsage::Stream &&
        device.GetDriverID() == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA) {
        anv_flags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    }
    u32 memory_type_bits = valid_memory_types;
    if (usage == MemoryUsage::Stream) {
        memory_type_bits = 0u;
    }
    const VmaAllocationCreateInfo alloc_ci = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | MemoryUsageVmaFlags(usage),
        .usage = MemoryUsageVma(usage),
        .requiredFlags = 0,
        .preferredFlags = MemoryUsagePreferredVmaFlags(usage) | anv_flags,
        .memoryTypeBits = memory_type_bits,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
        .priority = 0.f,
    };

    VkBuffer handle{};
    VmaAllocationInfo alloc_info{};
    VmaAllocation allocation{};
    VkMemoryPropertyFlags property_flags{};

    VkResult created = vmaCreateBufferWithAlignment(allocator, &ci, &alloc_ci, min_alignment, &handle,
                                           &allocation, &alloc_info);
    if (created != VK_SUCCESS) {
        ::Eden::Performance::graphics_memory_refusals.fetch_add(1, std::memory_order_relaxed);
        VmaAllocationCreateInfo outside_budget = alloc_ci;
        outside_budget.flags &= ~static_cast<VmaAllocationCreateFlags>(VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT);
        created = vmaCreateBufferWithAlignment(allocator, &ci, &outside_budget, min_alignment, &handle,
                                           &allocation, &alloc_info);
        // As the unaligned path: release guest memory and try again before giving up. The
        // aligned overload is used for sparse-compatible buffer cache buffers, which can be
        // large enough to hit the pool's budget on their own.
        created = RecoverDeviceMemory(created, ci.size, [&] {
            return vmaCreateBufferWithAlignment(allocator, &ci, &outside_budget, min_alignment,
                                                &handle, &allocation, &alloc_info);
        });
        static std::atomic<unsigned long long> refused{0};
        const auto times = refused.fetch_add(1, std::memory_order_relaxed) + 1;
        if ((times & (times - 1)) == 0) { // the 1st, 2nd, 4th, 8th ... time
            std::fprintf(stderr, "EDEN_GRAPHICS_MEMORY budget refused %llu requests; asked again outside it: result %d, buffer of %llu bytes, usage 0x%x\n",
                         times, int(created), static_cast<unsigned long long>(ci.size), unsigned(ci.usage));
        }
        if (created != VK_SUCCESS) {
            VmaBudget held[VK_MAX_MEMORY_HEAPS]{};
            vmaGetHeapBudgets(allocator, held);
            unsigned long long blocks = 0, used = 0;
            for (const VmaBudget& heap : held) {
                blocks += heap.statistics.blockBytes;
                used += heap.statistics.allocationBytes;
            }
            const auto free_block = ::Eden::Performance::graphics_memory_free.load(std::memory_order_relaxed);
            std::size_t guest_span = 0, guest_held = 0;
            ::Common::GuestMemoryUsage(&guest_span, &guest_held);
            std::fprintf(stderr, "EDEN_GRAPHICS_MEMORY refused outside the budget too (result %d): the pool's largest free "
                         "block is %llu MiB; the allocator holds %llu MiB in blocks, %llu MiB of it in use; the guest's "
                         "memory holds %llu MiB, the heap %llu MiB\n",
                         int(created), free_block ? free_block() >> 20 : 0ull, blocks >> 20, used >> 20,
                         static_cast<unsigned long long>(guest_held >> 20),
                         static_cast<unsigned long long>(eden_heap_committed() >> 20));
            ::Eden::Performance::ReportHeap("a buffer was refused");
        }
    }
    vk::Check(created);
    vmaGetAllocationMemoryProperties(allocator, allocation, &property_flags);

    u8 *data = reinterpret_cast<u8 *>(alloc_info.pMappedData);
    std::span<u8> mapped_data{};
    if (data) {
        mapped_data = std::span<u8>{data, ci.size};
    }
    const bool is_coherent = (property_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

    const vk::MemoryLocation location{
        .memory = alloc_info.deviceMemory,
        .offset = alloc_info.offset,
        .memory_type = alloc_info.memoryType,
    };
    return vk::Buffer(handle, *device.GetLogical(), allocator, allocation, mapped_data, is_coherent,
                      location, device.GetDispatchLoader());
}

MemoryCommit MemoryAllocator::Commit(const VkMemoryRequirements &reqs, MemoryUsage usage)
{
    const auto vma_usage = MemoryUsageVma(usage);
    VmaAllocationCreateInfo ci{};
    ci.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | MemoryUsageVmaFlags(usage);
    ci.usage = vma_usage;
    ci.memoryTypeBits = reqs.memoryTypeBits & valid_memory_types;
    ci.requiredFlags = 0;
    ci.preferredFlags = MemoryUsagePreferredVmaFlags(usage);

    VmaAllocation a{};
    VmaAllocationInfo info{};

    VkResult res = vmaAllocateMemory(allocator, &reqs, &ci, &a, &info);

    if (res != VK_SUCCESS) {
        // Relax 1: drop budget constraint
        auto ci2 = ci;
        ci2.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        res = vmaAllocateMemory(allocator, &reqs, &ci2, &a, &info);

        // Relax 2: if we preferred DEVICE_LOCAL, drop that preference
        if (res != VK_SUCCESS && (ci.preferredFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            auto ci3 = ci2;
            ci3.preferredFlags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            res = vmaAllocateMemory(allocator, &reqs, &ci3, &a, &info);
        }
        res = RecoverDeviceMemory(res, reqs.size,
                                  [&] { return vmaAllocateMemory(allocator, &reqs, &ci2, &a, &info); });
    }

    vk::Check(res);
    return MemoryCommit(allocator, a, info);
}

MemoryCommit MemoryAllocator::Commit(const vk::Buffer &buffer, MemoryUsage usage) {
    // Allocate memory appropriate for this buffer automatically
    const auto vma_usage = MemoryUsageVma(usage);

    VmaAllocationCreateInfo ci{};
    ci.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | MemoryUsageVmaFlags(usage);
    ci.usage = vma_usage;
    ci.requiredFlags = 0;
    ci.preferredFlags = MemoryUsagePreferredVmaFlags(usage);
    ci.pool = VK_NULL_HANDLE;
    ci.pUserData = nullptr;
    ci.priority = 0.0f;

    const VkBuffer raw = *buffer;

    VmaAllocation a{};
    VmaAllocationInfo info{};

    // Let VMA infer memory requirements from the buffer
    VkResult res = vmaAllocateMemoryForBuffer(allocator, raw, &ci, &a, &info);

    if (res != VK_SUCCESS) {
        auto ci2 = ci;
        ci2.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        res = vmaAllocateMemoryForBuffer(allocator, raw, &ci2, &a, &info);

        if (res != VK_SUCCESS && (ci.preferredFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            auto ci3 = ci2;
            ci3.preferredFlags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            res = vmaAllocateMemoryForBuffer(allocator, raw, &ci3, &a, &info);
        }
        const VkMemoryRequirements requirements =
            device.GetLogical().GetBufferMemoryRequirements(raw);
        res = RecoverDeviceMemory(res, requirements.size, [&] {
            return vmaAllocateMemoryForBuffer(allocator, raw, &ci2, &a, &info);
        });
    }

    vk::Check(res);
    vk::Check(vmaBindBufferMemory2(allocator, a, 0, raw, nullptr));
    return MemoryCommit(allocator, a, info);
}

#ifdef _WIN32
namespace {
/// The DXGI adapter of the Vulkan device (matched by LUID), or null. Looked up again when the
/// device changes (another GPU picked between sessions).
IDXGIAdapter3* FindDxgiAdapter(const Device& device) {
    VkPhysicalDeviceIDProperties id{};
    id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &id;
    device.GetPhysical().GetProperties2(properties);
    if (!id.deviceLUIDValid) {
        return nullptr;
    }
    LUID luid{};
    std::memcpy(&luid, id.deviceLUID, sizeof(luid));
    static LUID cached_luid{};
    static IDXGIAdapter3* cached{};
    if (cached != nullptr && cached_luid.LowPart == luid.LowPart &&
        cached_luid.HighPart == luid.HighPart) {
        return cached;
    }
    if (cached != nullptr) {
        cached->Release();
        cached = nullptr;
    }
    cached_luid = luid;
    const HMODULE dxgi = LoadLibraryW(L"dxgi.dll");
    if (dxgi == nullptr) {
        return nullptr;
    }
    using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
    const auto create =
        reinterpret_cast<CreateFactory>(reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1")));
    IDXGIFactory4* factory{};
    if (create == nullptr ||
        FAILED(create(__uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory)))) {
        return nullptr;
    }
    if (FAILED(factory->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter3),
                                          reinterpret_cast<void**>(&cached)))) {
        cached = nullptr;
    }
    factory->Release();
    return cached;
}
} // Anonymous namespace
#endif

void LogMemoryTrace(const Device& device) {
    const Common::ProcessMemoryUsage process = Common::GetProcessMemoryUsage();
    LOG_INFO(Render_Vulkan,
             "Memory trace: process commit {} MiB, working set {} MiB (private {} MiB), system "
             "commit {} of {} MiB",
             process.commit >> 20, process.working_set >> 20, process.private_working_set >> 20,
             process.system_commit >> 20, process.system_commit_limit >> 20);
#ifdef _WIN32
    // This process's use of the card's memory (local) and of system memory for the GPU
    // (non-local: host memory types, and what the OS moved out of video memory).
    IDXGIAdapter3* const adapter = FindDxgiAdapter(device);
    DXGI_QUERY_VIDEO_MEMORY_INFO local{};
    DXGI_QUERY_VIDEO_MEMORY_INFO non_local{};
    if (adapter != nullptr &&
        SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local)) &&
        SUCCEEDED(
            adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &non_local))) {
        LOG_INFO(Render_Vulkan,
                 "Memory trace: DXGI local {} of {} MiB, non-local {} of {} MiB",
                 local.CurrentUsage >> 20, local.Budget >> 20, non_local.CurrentUsage >> 20,
                 non_local.Budget >> 20);
    }
#endif
    const VmaAllocator allocator = device.GetAllocator();
    const VkPhysicalDeviceMemoryProperties* properties{};
    vmaGetMemoryProperties(allocator, &properties);
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
    vmaGetHeapBudgets(allocator, budgets.data());
    std::string heaps;
    for (u32 heap = 0; heap < properties->memoryHeapCount; ++heap) {
        const VmaBudget& budget = budgets[heap];
        fmt::format_to(std::back_inserter(heaps), "{}{} ({}): driver {} of {} MiB, VMA {}/{} MiB",
                       heap == 0 ? "" : "; ", heap,
                       (properties->memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0
                           ? "local"
                           : "host",
                       budget.usage >> 20, budget.budget >> 20,
                       budget.statistics.allocationBytes >> 20,
                       budget.statistics.blockBytes >> 20);
    }
    LOG_INFO(Render_Vulkan, "Memory trace: heaps {}", heaps);
    VmaTotalStatistics stats{};
    vmaCalculateStatistics(allocator, &stats);
    std::string types;
    for (u32 type = 0; type < properties->memoryTypeCount; ++type) {
        const VmaStatistics& s = stats.memoryType[type].statistics;
        if (s.blockCount == 0) {
            continue;
        }
        fmt::format_to(std::back_inserter(types), "{}{} (heap {}, flags {:#x}): {} allocs {}/{} MiB",
                       types.empty() ? "" : "; ", type, properties->memoryTypes[type].heapIndex,
                       static_cast<u32>(properties->memoryTypes[type].propertyFlags),
                       s.allocationCount, s.allocationBytes >> 20, s.blockBytes >> 20);
    }
    LOG_INFO(Render_Vulkan, "Memory trace: types {}", types);
}

void PrintMemoryStatistics(const Device& device) {
    const VmaAllocator allocator = device.GetAllocator();
    const VkPhysicalDeviceMemoryProperties* properties{};
    vmaGetMemoryProperties(allocator, &properties);
    VmaTotalStatistics stats{};
    vmaCalculateStatistics(allocator, &stats);
    for (u32 type = 0; type < properties->memoryTypeCount; ++type) {
        const VmaStatistics& s = stats.memoryType[type].statistics;
        if (s.blockCount == 0) {
            continue;
        }
        std::fprintf(stderr, "EDEN_MEM_SPLIT vma type=%u flags=0x%x blocks=%u/%lluMiB allocations=%u/%lluMiB\n",
                     type, static_cast<unsigned>(properties->memoryTypes[type].propertyFlags), s.blockCount,
                     static_cast<unsigned long long>(s.blockBytes >> 20), s.allocationCount,
                     static_cast<unsigned long long>(s.allocationBytes >> 20));
    }
}

void LogVmaPools(const Device& device) {
    // One line per 10 s, at most: the caches ask for memory in use every frame (Device::
    // GetDeviceMemoryUsage), which is where this is called from.
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_VK_POOLS");
        return value == nullptr || value[0] != '0';
    }();
    static std::atomic<long long> next_ns{0};
    const long long now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();
    if (!enabled || now < next_ns.load(std::memory_order_relaxed)) {
        return;
    }
    next_ns.store(now + 10'000'000'000, std::memory_order_relaxed);
    const VmaAllocator allocator = device.GetAllocator();
    const VkPhysicalDeviceMemoryProperties* properties{};
    vmaGetMemoryProperties(allocator, &properties);
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
    vmaGetHeapBudgets(allocator, budgets.data());
    u64 blocks = 0, used = 0;
    for (u32 heap = 0; heap < properties->memoryHeapCount; ++heap) {
        blocks += budgets[heap].statistics.blockBytes;
        used += budgets[heap].statistics.allocationBytes;
    }
    if (blocks == 0) {
        return; // nothing allocated yet
    }
    // Where the empty space is: the pools of their own, and the default blocks as what is left
    // (the heap figures include both). Empty is blocks less allocations: a block goes back to the
    // driver only once every allocation in it is gone, which is how Ocarina held 0.85 GB of its
    // 4.35 GB empty while the pool was at 0 for minutes (2026-11-16).
    u64 buffer_blocks = 0, buffer_used = 0, sparse_blocks = 0, sparse_used = 0;
    device.SeparatePoolUse(Device::SeparatePool::DeviceBuffers, buffer_blocks, buffer_used);
    device.SeparatePoolUse(Device::SeparatePool::SparsePages, sparse_blocks, sparse_used);
    const u64 pool_blocks = buffer_blocks + sparse_blocks;
    const u64 pool_used = buffer_used + sparse_used;
    const u64 default_blocks = blocks > pool_blocks ? blocks - pool_blocks : 0;
    const u64 default_used = used > pool_used ? used - pool_used : 0;
    LOG_INFO(Render_Vulkan,
             "EDEN_VK_POOLS empty space: default blocks {} MiB of {} MiB, device buffers {} MiB of "
             "{} MiB, sparse pages {} MiB of {} MiB",
             (default_blocks - std::min(default_blocks, default_used)) >> 20, default_blocks >> 20,
             (buffer_blocks - std::min(buffer_blocks, buffer_used)) >> 20, buffer_blocks >> 20,
             (sparse_blocks - std::min(sparse_blocks, sparse_used)) >> 20, sparse_blocks >> 20);
}

void LogMemoryStatistics(const Device& device) {
    const VmaAllocator allocator = device.GetAllocator();
    const VkPhysicalDeviceMemoryProperties* properties{};
    vmaGetMemoryProperties(allocator, &properties);
    VmaTotalStatistics stats{};
    vmaCalculateStatistics(allocator, &stats);
    for (u32 type = 0; type < properties->memoryTypeCount; ++type) {
        const VmaStatistics& s = stats.memoryType[type].statistics;
        if (s.blockCount == 0) {
            continue;
        }
        LOG_DEBUG(Render_Vulkan, "VMA type {} (heap {}, flags {:#x}): {} blocks {} MiB, {} allocations {} MiB",
                 type, properties->memoryTypes[type].heapIndex,
                 static_cast<u32>(properties->memoryTypes[type].propertyFlags), s.blockCount,
                 s.blockBytes >> 20, s.allocationCount, s.allocationBytes >> 20);
    }
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
    vmaGetHeapBudgets(allocator, budgets.data());
    for (u32 heap = 0; heap < properties->memoryHeapCount; ++heap) {
        LOG_INFO(Render_Vulkan, "VMA heap {}: usage {} MiB of budget {} MiB, VMA blocks {} MiB",
                 heap, budgets[heap].usage >> 20, budgets[heap].budget >> 20,
                 budgets[heap].statistics.blockBytes >> 20);
    }
}

} // namespace Vulkan
