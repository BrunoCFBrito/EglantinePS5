#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Builds compiler-rt's profile runtime for x86_64-sie-ps5, the target this app compiles for
# (headless/ps5.cmake). Clang ships that runtime only for its own host and the Payload SDK has
# none, so EDEN_PGO builds link this archive (patches/pgo-plan-2026-11-21.md):
#   tools/build-pgo-runtime.sh <compiler-rt-18.1.8 source> <output .a>
set -euo pipefail
export PATH=${EDEN_TOOL_PATH:+$EDEN_TOOL_PATH:}/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_dir=${1:?usage: tools/build-pgo-runtime.sh <compiler-rt source> <output .a>}
output=${2:?usage: tools/build-pgo-runtime.sh <compiler-rt source> <output .a>}
sdk=${PS5_PAYLOAD_SDK:-$(cat "$root/.local/headless-cache")/sdk}
profile="$source_dir/lib/profile"
test -f "$profile/InstrProfilingFile.c"
test -d "$source_dir/include/profile"
test -f "$sdk/target/lib/libSceLibcInternal.so"
work="${output%.a}.objects"
rm -rf "$work"
mkdir -p "$work" "$(dirname "$output")"
# The target's own SDK headers, no glibc: no fortify __*_chk calls and no __stack_chk_fail. No
# COMPILER_RT_HAS_UNAME (libkernel has no uname; the runtime's fallback is gethostname) and no
# COMPILER_RT_HAS_FCNTL_LCK (one thread writes at a time under the hook's mutex).
cflags=(-target x86_64-sie-ps5 -O2 -g0 -isysroot "$sdk"
        -isystem "$sdk/target/include" -fno-stack-protector -fno-plt
        -ffunction-sections -fdata-sections -fno-profile-generate -fno-profile-instr-generate
        -DCOMPILER_RT_HAS_ATOMICS=1 -DNDEBUG -U__FreeBSD__ -I "$profile" -I "$source_dir/include/profile")
# -U__FreeBSD__: the PS5 target predefines __FreeBSD__ (9), which would select procctl(2) and
# __assert paths the SDK does not export. Without it the runtime takes its portable branches.
cxxflags=("${cflags[@]}" -isystem "$sdk/target/include/c++/v1" -fno-rtti -fexceptions)
sources=(InstrProfiling.c InstrProfilingBuffer.c InstrProfilingFile.c InstrProfilingInternal.c
         InstrProfilingMerge.c InstrProfilingMergeFile.c InstrProfilingNameVar.c
         InstrProfilingUtil.c InstrProfilingValue.c InstrProfilingVersionVar.c
         InstrProfilingWriter.c)
for name in "${sources[@]}"; do
    clang-18 "${cflags[@]}" -c "$profile/$name" -o "$work/${name%.c}.o"
done
# The PS5 platform layer (section boundaries; the stock Linux layer needs dl_iterate_phdr and the
# stock catch-all layer never sees the compiler's section-based counters).
clang-18 "${cflags[@]}" -c "$root/tools/pgo-platform-ps5.c" -o "$work/pgo-platform-ps5.o"
clang++-18 "${cxxflags[@]}" -c "$profile/InstrProfilingRuntime.cpp" -o "$work/InstrProfilingRuntime.o"
llvm-ar-18 rcs "$output" "$work"/*.o
echo "== Built $output"
# __start_/__stop___llvm_prf_* are synthesised by the linker from the instrumented objects.
llvm-nm-18 --undefined-only "$output" | awk '{print $2}' | sort -u
