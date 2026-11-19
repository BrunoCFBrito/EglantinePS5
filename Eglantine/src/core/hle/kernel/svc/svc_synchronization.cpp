// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <thread>

#include "common/scope_exit.h"
#include "common/scratch_buffer.h"
#include "core/core.h"
#include "core/hle/kernel/k_hardware_timer.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_readable_event.h"
#include "core/hle/kernel/svc.h"
#include "core/hle/kernel/svc_results.h"
#include "core/hle/service/kernel_helpers.h"
#include "common/cpu_features.h"

namespace Kernel::Svc {

/// Close a handle
Result CloseHandle(Core::System& system, Handle handle) {
    LOG_TRACE(Kernel_SVC, "Closing handle {:#08x}", handle);

    // Remove the handle.
    R_UNLESS(GetCurrentProcess(system.Kernel()).GetHandleTable().Remove(system.Kernel(), handle),
             ResultInvalidHandle);

    R_SUCCEED();
}

/// Clears the signaled state of an event or process.
Result ResetSignal(Core::System& system, Handle handle) {
    LOG_TRACE(Kernel_SVC, "called handle {:#08x}", handle);

    // Get the current handle table.
    const auto& handle_table = GetCurrentProcess(system.Kernel()).GetHandleTable();

    // Try to reset as readable event.
    {
        KScopedAutoObject readable_event = handle_table.GetObject<KReadableEvent>(system.Kernel(), handle);
        if (readable_event.IsNotNull()) {
            R_RETURN(readable_event->Reset(system.Kernel()));
        }
    }

    // Try to reset as process.
    {
        KScopedAutoObject process = handle_table.GetObject<KProcess>(system.Kernel(), handle);
        if (process.IsNotNull()) {
            R_RETURN(process->Reset(system.Kernel()));
        }
    }

    R_THROW(ResultInvalidHandle);
}

/// Wait for the given handles to synchronize, timeout after the specified nanoseconds
Result WaitSynchronization(Core::System& system, int32_t* out_index, u64 user_handles,
                           int32_t num_handles, int64_t timeout_ns) {
    LOG_TRACE(Kernel_SVC, "called user_handles={:#x}, num_handles={}, timeout_ns={}", user_handles,
              num_handles, timeout_ns);

    // EDEN_NX2_WAIT_EARLY=<n>: up to n waits a second that return before blocking, with the
    // argument that failed. A loop retrying a rejected wait otherwise drains a core silently: the
    // call is 0x18 in the per-core histogram, but nothing says why it never blocks.
    static const unsigned wait_early_rate = [] {
        const char* value = std::getenv("EDEN_NX2_WAIT_EARLY");
        return value != nullptr ? static_cast<unsigned>(std::strtoul(value, nullptr, 10)) : 0u;
    }();
    const auto early_fail = [&](const char* why) {
        if (wait_early_rate == 0) {
            return;
        }
        static std::atomic<u64> last_second{0};
        static std::atomic<unsigned> used{0};
        const u64 second = static_cast<u64>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
        if (last_second.exchange(second, std::memory_order_relaxed) != second) {
            used.store(0, std::memory_order_relaxed);
        }
        if (used.fetch_add(1, std::memory_order_relaxed) < wait_early_rate) {
            LOG_INFO(Kernel_SVC, "WAIT_EARLY t{} why={} user={:#x} count={} timeout_ns={}",
                     GetCurrentThread(system.Kernel()).GetThreadId(), why, user_handles, num_handles,
                     timeout_ns);
        }
    };
    // EDEN_NX2_INVALID_WAIT_SLEEP_US: an experiment, not a fix. A guest loop that retries a
    // rejected wait burns its host core at a few hundred thousand calls a second; this pauses
    // each rejected call, which measures what freeing that core is worth before the cause is known.
    static const unsigned invalid_wait_sleep_us = [] {
        const char* value = std::getenv("EDEN_NX2_INVALID_WAIT_SLEEP_US");
        return value != nullptr ? static_cast<unsigned>(std::strtoul(value, nullptr, 10)) : 0u;
    }();

    // Ensure number of handles is valid.
    if (num_handles < 0 || num_handles > Svc::ArgumentHandleCountMax) {
        early_fail("range");
        R_THROW(ResultOutOfRange);
    }

    // Get the synchronization context.
    auto& handle_table = GetCurrentProcess(system.Kernel()).GetHandleTable();
    auto objs = GetCurrentThread(system.Kernel()).GetSynchronizationObjectBuffer();
    auto handles = GetCurrentThread(system.Kernel()).GetHandleBuffer();

    // Copy user handles.
    if (num_handles > 0) {
        // Get the handles.
        if (!GetCurrentMemory(system.Kernel()).ReadBlock(user_handles, handles.data(), sizeof(Handle) * num_handles)) {
            early_fail("pointer");
            R_THROW(ResultInvalidPointer);
        }

        // Convert the handles to objects.
        if (!handle_table.GetMultipleObjects<KSynchronizationObject>(system.Kernel(), objs.data(), handles.data(), num_handles)) {
            early_fail("handle");
            if (invalid_wait_sleep_us != 0) {
                std::this_thread::sleep_for(std::chrono::microseconds{invalid_wait_sleep_us});
            }
            R_THROW(ResultInvalidHandle);
        }
    }

    // Ensure handles are closed when we're done.
    SCOPE_EXIT {
        for (auto i = 0; i < num_handles; ++i) {
            objs[i]->Close(system.Kernel());
        }
    };

    // Convert the timeout from nanoseconds to ticks.
    s64 timeout;
    if (timeout_ns > 0) {
        u64 ticks = system.Kernel().HardwareTimer().GetTick();
        ticks += timeout_ns;
        ticks += 2;

        timeout = ticks;
    } else {
        timeout = timeout_ns;
    }

    // EDEN_NX2_WAIT_RESULT=<budget>[:<first_present>]: log what a bounded number of waits returned
    // and how long they took, so an instantly-returning wait can be told from a blocked one.
    static const std::pair<unsigned, u64> wait_result = [] {
        const char* text = std::getenv("EDEN_NX2_WAIT_RESULT");
        if (text == nullptr) return std::pair<unsigned, u64>{0u, 0ull};
        const char* colon = std::strchr(text, ':');
        const unsigned budget = static_cast<unsigned>(std::strtoul(text, nullptr, 10));
        const u64 first = colon != nullptr ? std::strtoull(colon + 1, nullptr, 10) : 0ull;
        return std::pair<unsigned, u64>{budget, first};
    }();
    const long long wait_begin =
        wait_result.first != 0 ? Common::g_wall_clock.GetTimeNS().count() : 0;
    // Wait on the objects.
    Result res = KSynchronizationObject::Wait(system.Kernel(), out_index, objs.data(), num_handles, timeout);
    if (wait_result.first != 0) {
        const u64 presents = Core::g_nx2_present_count.load(std::memory_order_relaxed);
        static std::atomic<unsigned> used{0};
        if (presents >= wait_result.second &&
            used.fetch_add(1, std::memory_order_relaxed) < wait_result.first) {
            std::string list;
            char text[96];
            for (int i = 0; i < num_handles; ++i) {
                KScopedAutoObject object =
                    handle_table.GetObject<KSynchronizationObject>(system.Kernel(), handles[i]);
                const char* type = object.IsNotNull() ? object->GetTypeName() : "invalid";
                if (object.IsNotNull() && std::strcmp(type, "KReadableEvent") == 0) {
                    KScopedAutoObject<KReadableEvent> event =
                        handle_table.GetObject<KReadableEvent>(system.Kernel(), handles[i]);
                    std::snprintf(text, sizeof(text), " %#x=%s(%s)", handles[i], type,
                                  event.IsNotNull()
                                      ? Service::KernelHelpers::DebugEventName(
                                            event.GetPointerUnsafe())
                                            .c_str()
                                      : "?");
                } else {
                    std::snprintf(text, sizeof(text), " %#x=%s", handles[i], type);
                }
                list += text;
            }
            const long long elapsed =
                Common::g_wall_clock.GetTimeNS().count() - wait_begin;
            LOG_INFO(Kernel_SVC,
                     "WAIT_RESULT t{} presents={} timeout_ns={} result={:#x} elapsed_us={} "
                     "handles={}{}",
                     GetCurrentThread(system.Kernel()).GetThreadId(), presents, timeout_ns,
                     res.raw, elapsed / 1000, num_handles, list);
        }
    }

    R_SUCCEED_IF(res == ResultSessionClosed);
    R_RETURN(res);
}

/// Resumes a thread waiting on WaitSynchronization
Result CancelSynchronization(Core::System& system, Handle handle) {
    LOG_TRACE(Kernel_SVC, "called handle={:#x}", handle);

    // Get the thread from its handle.
    KScopedAutoObject thread = GetCurrentProcess(system.Kernel()).GetHandleTable().GetObject<KThread>(system.Kernel(), handle);
    R_UNLESS(thread.IsNotNull(), ResultInvalidHandle);

    // Cancel the thread's wait.
    thread->WaitCancel(system.Kernel());
    R_SUCCEED();
}

void SynchronizePreemptionState(Core::System& system) {
    // Lock the scheduler.
    KScopedSchedulerLock sl{system.Kernel()};

    // If the current thread is pinned, unpin it.
    KProcess* cur_process = GetCurrentProcessPointer(system.Kernel());
    const auto core_id = GetCurrentCoreId(system.Kernel());

    if (cur_process->GetPinnedThread(core_id) == GetCurrentThreadPointer(system.Kernel())) {
        // Clear the current thread's interrupt flag.
        GetCurrentThread(system.Kernel()).ClearInterruptFlag(system.Kernel());

        // Unpin the current thread.
        cur_process->UnpinCurrentThread(system.Kernel());
    }
}

Result CloseHandle64(Core::System& system, Handle handle) {
    R_RETURN(CloseHandle(system, handle));
}

Result ResetSignal64(Core::System& system, Handle handle) {
    R_RETURN(ResetSignal(system, handle));
}

Result WaitSynchronization64(Core::System& system, int32_t* out_index, uint64_t handles,
                             int32_t num_handles, int64_t timeout_ns) {
    R_RETURN(WaitSynchronization(system, out_index, handles, num_handles, timeout_ns));
}

Result CancelSynchronization64(Core::System& system, Handle handle) {
    R_RETURN(CancelSynchronization(system, handle));
}

void SynchronizePreemptionState64(Core::System& system) {
    SynchronizePreemptionState(system);
}

Result CloseHandle64From32(Core::System& system, Handle handle) {
    R_RETURN(CloseHandle(system, handle));
}

Result ResetSignal64From32(Core::System& system, Handle handle) {
    R_RETURN(ResetSignal(system, handle));
}

Result WaitSynchronization64From32(Core::System& system, int32_t* out_index, uint32_t handles,
                                   int32_t num_handles, int64_t timeout_ns) {
    R_RETURN(WaitSynchronization(system, out_index, handles, num_handles, timeout_ns));
}

Result CancelSynchronization64From32(Core::System& system, Handle handle) {
    R_RETURN(CancelSynchronization(system, handle));
}

void SynchronizePreemptionState64From32(Core::System& system) {
    SynchronizePreemptionState(system);
}

} // namespace Kernel::Svc
