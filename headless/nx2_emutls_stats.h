// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// EDEN_NX2_EMUTLS_STATS=1: how often the PS5 build's emulated TLS reaches
// __emutls_get_address, per thread and per call site. The build compiles every emulated-TLS unit
// with -femulated-tls, so each thread_local read is a call into compiler-rt. nx2_emutls_stats.cpp
// wraps that call (--wrap=__emutls_get_address) and prints one EDEN_EMUTLS line set every 10 s to
// stderr. Off unless the variable is set; the wrapper is a load and a branch when it is off.
namespace Eden::Nx2EmutlsStats {

/// Names the calling thread's counter slot. Called by Performance::RegisterWorker, which every
/// Common::SetCurrentThreadName goes through on PS5.
void NoteCurrentThread(const char* name);

} // namespace Eden::Nx2EmutlsStats
