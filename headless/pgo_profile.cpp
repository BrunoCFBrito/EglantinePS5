// SPDX-License-Identifier: GPL-3.0-or-later
#include "pgo_profile.h"

#if defined(EDEN_PGO)
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>

#include "storage_paths.h"

// compiler-rt lib/profile, built for x86_64-sie-ps5 by tools/build-pgo-runtime.sh.
extern "C" {
void __llvm_profile_set_filename(const char* name);
int __llvm_profile_write_file(void);
}
#endif

namespace Eden::Pgo {

#if defined(EDEN_PGO)
namespace {

std::mutex write_lock;
std::atomic<bool> periodic_started{};
std::string path;

std::string ProfilePath() {
    const char* const configured = std::getenv("EDEN_PGO_FILE");
    if (configured != nullptr && *configured != '\0') {
        return configured;
    }
    return LogsDir() + "/pgo.profraw";
}

} // Anonymous namespace
#endif

std::string Write() {
#if defined(EDEN_PGO)
    const std::lock_guard lock{write_lock};
    if (path.empty()) {
        path = ProfilePath();
    }
    __llvm_profile_set_filename(path.c_str());
    if (__llvm_profile_write_file() != 0) {
        std::fprintf(stderr, "PGO profile write failed: %s\n", path.c_str());
        return {};
    }
    return path;
#else
    return {};
#endif
}

void StartPeriodicWrites() {
#if defined(EDEN_PGO)
    if (periodic_started.exchange(true)) {
        return;
    }
    const char* const configured = std::getenv("EDEN_PGO_FLUSH_SECONDS");
    const int seconds = configured != nullptr ? std::atoi(configured) : 0;
    if (seconds <= 0 || seconds > 3600) {
        return;
    }
    std::thread{[seconds] {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds{seconds});
            const std::string written = Write();
            if (!written.empty()) {
                std::fprintf(stderr, "PGO profile written: %s\n", written.c_str());
            }
        }
    }}.detach();
#endif
}

} // namespace Eden::Pgo
