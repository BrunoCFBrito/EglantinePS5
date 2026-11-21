// SPDX-License-Identifier: GPL-3.0-or-later
// The instrumented (EDEN_PGO=generate) build's counters, written where the console's FTP server can
// take them (patches/pgo-plan-2026-11-21.md). Inert in every other build.
#pragma once

#include <string>

namespace Eden::Pgo {

// Writes the counters now. Returns the file written, or an empty string when this build has no
// instrumentation or the write failed.
std::string Write();

// EDEN_PGO_FLUSH_SECONDS=<n>: Write() every n seconds once the debug server starts, so counters
// survive an app that is killed. Inert without EDEN_PGO or the variable.
void StartPeriodicWrites();

} // namespace Eden::Pgo
