// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdlib>
#include <memory>
#include <type_traits>
#include <utility>

namespace Shader {

template <typename T>
    requires std::is_destructible_v<T>
class ObjectPool {
public:
    explicit ObjectPool(size_t chunk_size = 8192) : new_chunk_size{chunk_size} {
        node = &chunks.emplace_back(new_chunk_size);
    }

    template <typename... Args>
        requires std::is_constructible_v<T, Args...>
    [[nodiscard]] T* Create(Args&&... args) {
        return std::construct_at(Memory(), std::forward<Args>(args)...);
    }

    /// Destroys the objects and returns their memory; ReleaseContents keeps it for reuse
    void Free() {
        chunks.clear();
        chunks.shrink_to_fit();
        node = &chunks.emplace_back(new_chunk_size);
    }

    void ReleaseContents() {
        if (chunks.empty()) {
            return;
        }
        Chunk& root{chunks.front()};
        if (root.used_objects == root.num_objects) {
            // Root chunk has been filled, squash allocations into it
            const size_t total_objects{root.num_objects + new_chunk_size * (chunks.size() - 1)};
            // A pool whose squashed chunk would pass the cap returns its memory instead of
            // keeping its largest shader's worth for good, in every main, spare, worker and stage
            // pool (Fable 5.2 ranked this first for the ~1.2 GB of process heap above the game's
            // needs; BC-250 at 64 MiB: cold-compile peak 5.6 -> 3.7 GB, worst freeze 13 -> 6.9 s).
            // Free() needs one chunk of the pool's own to start again, so this is compile time
            // spent only on shaders whose IR passes the cap.
            if (const size_t cap{CapBytes()};
                cap != 0 && total_objects * sizeof(Storage) > cap) {
                Free();
                return;
            }
            chunks.clear();
            chunks.emplace_back(total_objects);
        } else {
            root.Release();
            chunks.resize(1);
        }
        chunks.shrink_to_fit();
        node = &chunks.front();
    }

private:
    /// The most memory one pool may keep between shaders, from EDEN_NX2_IR_POOL_CAP_MIB (0 or
    /// unset keeps everything). PS5 defaults it to 64 MiB: ReleaseContents otherwise retains the
    /// largest shader a pool ever translated, and the BC-250 (same APU as the console) measured
    /// the cold-compile heap peak drop by 1.9 GB with this cap. Every pool reads the setting once.
    static size_t CapBytes() {
        static const size_t cap = [] {
            if (const char* const text = std::getenv("EDEN_NX2_IR_POOL_CAP_MIB")) {
                return static_cast<size_t>(std::strtoull(text, nullptr, 10)) << 20;
            }
#if defined(__PROSPERO__) // PS5
            return size_t{64} << 20;
#else
            return size_t{0};
#endif
        }();
        return cap;
    }

    struct NonTrivialDummy {
        NonTrivialDummy() noexcept {}
    };

    union Storage {
        Storage() noexcept {}
        ~Storage() noexcept {}

        NonTrivialDummy dummy{};
        T object;
    };

    struct Chunk {
        explicit Chunk() = default;
        explicit Chunk(size_t size)
            : num_objects{size}, storage{std::make_unique<Storage[]>(size)} {}

        Chunk& operator=(Chunk&& rhs) noexcept {
            Release();
            used_objects = std::exchange(rhs.used_objects, 0);
            num_objects = std::exchange(rhs.num_objects, 0);
            storage = std::move(rhs.storage);
            return *this;
        }

        Chunk(Chunk&& rhs) noexcept
            : used_objects{std::exchange(rhs.used_objects, 0)},
              num_objects{std::exchange(rhs.num_objects, 0)}, storage{std::move(rhs.storage)} {}

        ~Chunk() {
            Release();
        }

        // Destroys the objects themselves: Storage's destructor is empty on purpose (the union
        // also holds unused slots), so destroying the Storage elements would skip T's
        // destructor and leak what the objects own (e.g. an IR::Inst's phi operand list).
        void Release() {
            for (size_t index = 0; index < used_objects; ++index) {
                std::destroy_at(&storage[index].object);
            }
            used_objects = 0;
        }

        size_t used_objects{};
        size_t num_objects{};
        std::unique_ptr<Storage[]> storage;
    };

    [[nodiscard]] T* Memory() {
        Chunk* const chunk{FreeChunk()};
        return &chunk->storage[chunk->used_objects++].object;
    }

    [[nodiscard]] Chunk* FreeChunk() {
        if (node->used_objects != node->num_objects) {
            return node;
        }
        node = &chunks.emplace_back(new_chunk_size);
        return node;
    }

    Chunk* node{};
    std::vector<Chunk> chunks;
    size_t new_chunk_size{};
};

} // namespace Shader
