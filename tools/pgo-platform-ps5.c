// SPDX-License-Identifier: GPL-3.0-or-later
// Platform layer of compiler-rt's profile runtime for x86_64-sie-ps5 (tools/build-pgo-runtime.sh).
// Clang lowers IR PGO for PS4/PS5 with linker-section discovery (no __llvm_profile_register_* calls),
// the same as ELF Linux. The stock Linux layer needs dl_iterate_phdr and the stock catch-all layer
// registers functions the compiler never emits calls for, so this file maps the four sections by
// the linker-synthesised __start_/__stop_ symbols. Weak references resolve to 0 when a section is
// absent (for example __llvm_prf_vnds without indirect-call profiling), giving an empty range.
#include <stddef.h>
#include <stdint.h>

#include "InstrProfiling.h"
#include "InstrProfilingInternal.h"

#define PS5_PROF_SECTION(name)                                                                     \
    extern char __start___llvm_prf_##name[] __attribute__((weak, visibility("hidden")));           \
    extern char __stop___llvm_prf_##name[] __attribute__((weak, visibility("hidden")))

PS5_PROF_SECTION(data);
PS5_PROF_SECTION(cnts);
PS5_PROF_SECTION(names);
PS5_PROF_SECTION(vnds);

COMPILER_RT_VISIBILITY const __llvm_profile_data *__llvm_profile_begin_data(void) {
    return (const __llvm_profile_data *)__start___llvm_prf_data;
}
COMPILER_RT_VISIBILITY const __llvm_profile_data *__llvm_profile_end_data(void) {
    return (const __llvm_profile_data *)__stop___llvm_prf_data;
}
COMPILER_RT_VISIBILITY const char *__llvm_profile_begin_names(void) {
    return __start___llvm_prf_names;
}
COMPILER_RT_VISIBILITY const char *__llvm_profile_end_names(void) {
    return __stop___llvm_prf_names;
}
COMPILER_RT_VISIBILITY char *__llvm_profile_begin_counters(void) {
    return __start___llvm_prf_cnts;
}
COMPILER_RT_VISIBILITY char *__llvm_profile_end_counters(void) {
    return __stop___llvm_prf_cnts;
}
// IR PGO emits no bitmap section. Not referencing __start_/__stop___llvm_prf_bits matters: the
// packager rejects any undefined symbol that the SDK does not export, weak or not.
COMPILER_RT_VISIBILITY char *__llvm_profile_begin_bitmap(void) {
    return NULL;
}
COMPILER_RT_VISIBILITY char *__llvm_profile_end_bitmap(void) {
    return NULL;
}
COMPILER_RT_VISIBILITY uint32_t *__llvm_profile_begin_orderfile(void) {
    return NULL;
}
COMPILER_RT_VISIBILITY ValueProfNode *__llvm_profile_begin_vnodes(void) {
    return (ValueProfNode *)__start___llvm_prf_vnds;
}
COMPILER_RT_VISIBILITY ValueProfNode *__llvm_profile_end_vnodes(void) {
    return (ValueProfNode *)__stop___llvm_prf_vnds;
}
COMPILER_RT_VISIBILITY ValueProfNode *CurrentVNode = (ValueProfNode *)__start___llvm_prf_vnds;
COMPILER_RT_VISIBILITY ValueProfNode *EndVNode = (ValueProfNode *)__stop___llvm_prf_vnds;

// No binary-id note on this target: the profile carries an empty binary-id section.
COMPILER_RT_VISIBILITY int __llvm_write_binary_ids(ProfDataWriter *Writer) {
    (void)Writer;
    return 0;
}

// Host name for the %h file-name pattern. uname is not in libkernel, so the runtime's own uname
// branch is off (COMPILER_RT_HAS_UNAME undefined); gethostname comes from libSceLibcInternal.
int gethostname(char *name, size_t len);
COMPILER_RT_VISIBILITY int lprofGetHostName(char *Name, int Len) {
    if (Len <= 0 || gethostname(Name, (size_t)Len) != 0) {
        return -1;
    }
    Name[Len - 1] = '\0';
    return 0;
}
