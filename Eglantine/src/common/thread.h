// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2013 Dolphin Emulator Project
// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stop_token>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#if !defined(_WIN32)
#include <pthread.h>
#endif
#include "common/common_types.h"
#include "common/polyfill_thread.h"
#include "common/unique_function.h"

namespace Common {

#if !defined(_WIN32)
/// A thread whose stack is smaller than the platform's default, for the service and worker threads
/// whose work is shallow: the HLE service loops, the vsync thread, the time worker and event
/// observer, cache writers. On the PS5 the Payload SDK takes 2 MiB of the pool shared with the GPU
/// for every thread's stack (pool budget, 2026-11-17: 49 stacks, 98 MiB), and the pool is what a
/// game runs out of first. The emulated CPU cores, the shader compiler and the GPU threads keep the
/// default stack and stay std::jthread.
///
/// A std::jthread-compatible handle: runs its function with a std::stop_token when the function
/// takes one, and the destructor requests the stop and joins. EDEN_WORKER_STACK_KIB (1024 by
/// default, 256 to 8192) sets the stack, for one A/B run without a rebuild.
class WorkerThread {
public:
    WorkerThread() noexcept = default;

    template <typename Function>
    explicit WorkerThread(Function&& function) {
        using Callable = std::decay_t<Function>;
        auto state = std::make_shared<State>();
        state->task = [fn = std::forward<Function>(function)](std::stop_token token) mutable {
            if constexpr (std::is_invocable_v<Callable&, std::stop_token>) {
                fn(std::move(token));
            } else {
                fn();
            }
        };
        auto* holder = new std::shared_ptr<State>{state};
        pthread_attr_t attributes;
        const int initialized = pthread_attr_init(&attributes);
        if (initialized != 0) {
            delete holder;
            throw std::system_error(initialized, std::generic_category(), "WorkerThread");
        }
        (void)pthread_attr_setstacksize(&attributes, StackSize());
        const int created = pthread_create(&thread, &attributes, &WorkerThread::Entry, holder);
        (void)pthread_attr_destroy(&attributes);
        if (created != 0) {
            delete holder;
            throw std::system_error(created, std::generic_category(), "WorkerThread");
        }
        state_ = std::move(state);
        joinable_ = true;
    }

    WorkerThread(const WorkerThread&) = delete;
    WorkerThread& operator=(const WorkerThread&) = delete;

    WorkerThread(WorkerThread&& other) noexcept
        : thread{other.thread}, state_{std::move(other.state_)}, joinable_{other.joinable_} {
        other.joinable_ = false;
    }

    WorkerThread& operator=(WorkerThread&& other) noexcept {
        if (this != &other) {
            if (joinable_) {
                request_stop();
                (void)pthread_join(thread, nullptr);
            }
            thread = other.thread;
            state_ = std::move(other.state_);
            joinable_ = other.joinable_;
            other.joinable_ = false;
        }
        return *this;
    }

    ~WorkerThread() {
        if (joinable_) {
            request_stop();
            (void)pthread_join(thread, nullptr);
        }
    }

    [[nodiscard]] bool joinable() const noexcept {
        return joinable_;
    }

    void join() {
        if (!joinable_) {
            throw std::system_error(std::make_error_code(std::errc::invalid_argument),
                                    "WorkerThread::join");
        }
        const int result = pthread_join(thread, nullptr);
        if (result != 0) {
            throw std::system_error(result, std::generic_category(), "WorkerThread::join");
        }
        joinable_ = false;
    }

    void detach() {
        if (!joinable_) {
            throw std::system_error(std::make_error_code(std::errc::invalid_argument),
                                    "WorkerThread::detach");
        }
        const int result = pthread_detach(thread);
        if (result != 0) {
            throw std::system_error(result, std::generic_category(), "WorkerThread::detach");
        }
        joinable_ = false;
    }

    bool request_stop() noexcept {
        return state_ != nullptr && state_->stop.request_stop();
    }

    [[nodiscard]] std::stop_token get_stop_token() const noexcept {
        return state_ != nullptr ? state_->stop.get_token() : std::stop_token{};
    }

    void swap(WorkerThread& other) noexcept {
        std::swap(thread, other.thread);
        state_.swap(other.state_);
        std::swap(joinable_, other.joinable_);
    }

private:
    struct State {
        std::stop_source stop;
        UniqueFunction<void, std::stop_token> task;
    };

    static std::size_t StackSize() {
        static const std::size_t size = [] {
            const char* const text = std::getenv("EDEN_WORKER_STACK_KIB");
            const long kib = text != nullptr ? std::strtol(text, nullptr, 10) : 1024;
            return static_cast<std::size_t>(kib < 256 ? 256 : (kib > 8192 ? 8192 : kib)) << 10;
        }();
        return size;
    }

    static void* Entry(void* opaque) {
        std::unique_ptr<std::shared_ptr<State>> holder{static_cast<std::shared_ptr<State>*>(opaque)};
        const std::shared_ptr<State> state = *holder;
        holder.reset();
        state->task(state->stop.get_token());
        return nullptr;
    }

    pthread_t thread{};
    std::shared_ptr<State> state_;
    bool joinable_{};
};
#else
/// Windows: the operating system's default stack, as before.
using WorkerThread = std::jthread;
#endif

class Event {
public:
#ifdef _WIN32
    ~Event();
#endif

    void Set() {
        std::scoped_lock lk{mutex};
        if (!is_set) {
            is_set = true;
#ifdef _WIN32
            if (sleepers != 0) {
                WakeSleeper();
            }
#endif
            condvar.notify_one();
        }
    }

    void Wait() {
        std::unique_lock lk{mutex};
        condvar.wait(lk, [&] { return is_set.load(); });
        is_set = false;
    }

    bool WaitFor(const std::chrono::nanoseconds time);

    /// Like WaitFor, but always blocks in the OS. On Windows, WaitFor keeps the thread running
    /// in a hardware wait; this sleeps on a high-resolution waitable timer instead.
    bool SleepFor(const std::chrono::nanoseconds time);

    template<class Clock, class Duration>
    bool WaitUntil(const std::chrono::time_point<Clock, Duration> time) {
        std::unique_lock lk{mutex};
        if (!condvar.wait_until(lk, time, [this] { return is_set.load(); }))
            return false;
        is_set = false;
        return true;
    }

    void Reset() {
        std::unique_lock lk{mutex};
        // no other action required, since wait loops on the predicate and any lingering signal will
        // get cleared on the first iteration
        is_set = false;
    }

    [[nodiscard]] bool IsSet() const {
        return is_set;
    }

private:
#ifdef _WIN32
    void WakeSleeper();
#endif

    alignas(64) std::atomic<bool> is_set{false};
    std::condition_variable condvar;
    std::mutex mutex;
#ifdef _WIN32
    void* sleep_wake{};
    u32 sleepers{};
#endif
};

class Barrier {
public:
    explicit Barrier(std::size_t count_) : count(count_) {}

    /// Blocks until all "count" threads have called Sync()
    bool Sync(std::stop_token token = {}) {
        std::unique_lock lk{mutex};
        const std::size_t current_generation = generation;

        if (++waiting == count) {
            generation++;
            waiting = 0;
            condvar.notify_all();
            return true;
        } else {
            condvar.wait(lk, token,
                        [this, current_generation] { return current_generation != generation; });
            return !token.stop_requested();
        }
    }

private:
    std::condition_variable_any condvar;
    std::mutex mutex;
    std::size_t count;
    std::size_t waiting = 0;
    std::size_t generation = 0; // Incremented once each time the barrier is used
};

enum class ThreadPriority : u32 {
    Low = 0,
    Normal = 1,
    High = 2,
    VeryHigh = 3,
    Critical = 4,
};

enum class ThreadPlacement : u32 {
    Default = 0,
    Background = 1,
    Efficiency = 2,
};

void SetCurrentThreadPriority(ThreadPriority new_priority);
void SetCurrentThreadName(const char* name);
void SetCurrentThreadToPerformanceCores();
void SetCurrentThreadToEfficiencyCores();
void SetCurrentThreadToBackgroundWork();
void SetCurrentThreadToAllCores();

enum class PinnedRole : u32 { GpuFront, GpuBack, GuestCore };

/// The experimental pin_gpu_threads setting (Windows and Linux): the split GPU thread's front and
/// back each get a physical performance core of their own and the guest cores keep off those two.
/// Skipped, with one log line, on CPUs with too few cores, with EDEN_HYBRID_CORES=0 (Windows), and
/// on other hosts.
void PinCurrentThread(PinnedRole role);

void RefreshThreadPolicies();
void PollThreadPolicies();

} // namespace Common
