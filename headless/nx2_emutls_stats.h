// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// EDEN_NX2_EMUTLS_STATS=1: how often the PS5 build's emulated TLS reaches
// __emutls_get_address, per thread and per call site. The build compiles every emulated-TLS unit
// with -femulated-tls, so each thread_local read is a call into compiler-rt. In a build made with
// EDEN_EMUTLS_STATS_BUILD=1, nx2_emutls_stats.cpp wraps that call (--wrap=__emutls_get_address)
// and prints one EDEN_EMUTLS line set every 10 s to stderr. That build is opt-in because the
// wrapper otherwise costs every thread_local read, switch off or not; a default build compiles
// the hook below to nothing.
#if defined(EDEN_EMUTLS_STATS)
namespace Eden::Nx2EmutlsStats {

/// Names the calling thread's counter slot. Called by Performance::RegisterWorker, which every
/// Common::SetCurrentThreadName goes through on PS5.
void NoteCurrentThread(const char* name);

} // namespace Eden::Nx2EmutlsStats
#else
namespace Eden::Nx2EmutlsStats {

/// No counter in this build: RegisterWorker's hook costs nothing.
inline void NoteCurrentThread(const char*) {}

} // namespace Eden::Nx2EmutlsStats
#endif
