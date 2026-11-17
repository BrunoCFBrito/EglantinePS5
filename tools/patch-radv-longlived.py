#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pipeline-lifetime RADV allocations come from Eden's shared long-lived heap arenas.

RADV allocates what lives as long as a pipeline (the driver's object for a shader, its CPU copy
of the code, VS prologs, the strings parsed out of a shader binary, descriptor set layouts,
pipeline cache objects) on the threads that build shaders, between the compiler's temporary
ralloc blocks. In a thread's Eden heap arena those objects keep the arena non-empty for the life
of the pipeline, so the emulator cannot give it back after a cold pipeline-cache build (231
arenas held 462 MiB with 83 MiB live). The objects now come from the shared long-lived arena set
(headless/heap_arenas.inc, eden_heap_long_lived_malloc/calloc) through weak symbols, so a RADV
build without the emulator's heap keeps plain malloc/calloc. Debug sections a shader binary
carries are parsed only when a debug option will read them, which keeps that memory from being
allocated at all. The matching frees stay free()/realloc(). Applied after the memory scripts;
patch-radv-pipeline-oom.py builds on the allocations this script introduces. Patches the build
export, never the Mesa checkout.
"""
from pathlib import Path
import hashlib
import sys

root = Path(__file__).resolve().parents[1]
fingerprint = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
if sys.argv[1:2] == ['--check']:
    stamp = Path(sys.argv[2])
    raise SystemExit(0 if stamp.is_file() and stamp.read_text().strip() == fingerprint else 1)
if sys.argv[1:2] == ['--digest']:
    print(fingerprint)
    raise SystemExit(0)

HEADER = r'''/*
 * Copyright © 2026 Eden PS5 port
 * SPDX-License-Identifier: MIT
 */

#ifndef VK_EDEN_HEAP_H
#define VK_EDEN_HEAP_H

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* The Eden PS5 emulator's shared long-lived arena set (headless/heap_arenas.inc).
 *
 * RADV allocates what lives as long as a pipeline (the driver's object for a shader, its CPU copy
 * of the code, VS prologs, the strings parsed out of a shader binary, descriptor set layouts) on
 * the threads that build shaders, between the compiler's temporary ralloc blocks. In a thread's
 * arena those objects keep the arena non-empty for the life of the pipeline, so the emulator
 * cannot give the arena back after a cold pipeline-cache build (231 arenas held 462 MiB with
 * 83 MiB live). These entry points put such objects in arenas shared by all threads and never
 * given back, so the compiler's blocks stay in the thread's arena and it empties after the build.
 *
 * The symbols are weak: a RADV build without the Eden heap keeps plain malloc/calloc. free()
 * needs no counterpart: an Eden long-lived block is an ordinary block of the emulator's heap,
 * and free()/realloc() find the block's arena by address (vk_eden_heap.h callers only choose
 * where the object is allocated; every free path stays free()).
 */
void *eden_heap_long_lived_malloc(size_t size) __attribute__((weak));
void *eden_heap_long_lived_calloc(size_t count, size_t size) __attribute__((weak));
void eden_heap_long_lived_free(void *address) __attribute__((weak));

static inline void *
radv_ps5_long_lived_malloc(size_t size)
{
   return eden_heap_long_lived_malloc ? eden_heap_long_lived_malloc(size) : malloc(size);
}

static inline void *
radv_ps5_long_lived_calloc(size_t count, size_t size)
{
   return eden_heap_long_lived_calloc ? eden_heap_long_lived_calloc(count, size) : calloc(count, size);
}

static inline char *
radv_ps5_long_lived_strdup(const char *string)
{
   if (string == NULL)
      return NULL;

   const size_t size = strlen(string) + 1;
   char *copy = radv_ps5_long_lived_malloc(size);
   if (copy != NULL)
      memcpy(copy, string, size);

   return copy;
}

static inline void
radv_ps5_long_lived_free(void *address)
{
   if (eden_heap_long_lived_free)
      eden_heap_long_lived_free(address);
   else
      free(address);
}

#endif /* VK_EDEN_HEAP_H */
'''

KEEP_DEBUG_INFO = r'''/* Eden PS5: whether anything will read the debug sections a shader binary carries (disasm, IR,
 * statistics, instruction debug info). Without a debug option none are produced (ACO records
 * them only for a shader dump or for executable info), and parsing them into the shader's
 * pipeline-lifetime dbg would only spend memory and pin the compile thread's arena; with one,
 * skipping the parse would empty the dumps and the executable-property queries. */
bool
radv_keep_binary_debug_info(const struct radv_device *device)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);
   const struct radv_instance *instance = radv_physical_device_instance(pdev);

   /* Fault detection (RADV_DEBUG=hang) and the trap handler keep shader info for the dumps. */
   if (device->keep_shader_info)
      return true;

   /* Shader dumps and statistics (radv_shader_dump_asm, radv_dump_shader_stats, the prolog and
    * epilog dumps), and RADV_DEBUG=shaders, which captures executable info. */
   if (instance->debug_flags &
       (RADV_DEBUG_DUMP_SHADERS | RADV_DEBUG_DUMP_VS | RADV_DEBUG_DUMP_TCS | RADV_DEBUG_DUMP_TES |
        RADV_DEBUG_DUMP_GS | RADV_DEBUG_DUMP_PS | RADV_DEBUG_DUMP_TASK | RADV_DEBUG_DUMP_MESH |
        RADV_DEBUG_DUMP_CS | RADV_DEBUG_DUMP_META_SHADERS | RADV_DEBUG_DUMP_ASM | RADV_DEBUG_DUMP_SHADER_STATS |
        RADV_DEBUG_DUMP_PROLOGS | RADV_DEBUG_DUMP_EPILOGS | RADV_DEBUG_DUMP_TRAP_HANDLER | RADV_DEBUG_PSO_HISTORY))
      return true;

   /* RGP captures shader statistics to correlate shader hashes. */
   if (instance->vk.trace_mode & RADV_TRACE_MODE_RGP)
      return true;

   /* VK_KHR_pipeline_executable_properties hands the NIR/IR/disasm to the application. */
   if (device->vk.enabled_extensions.KHR_pipeline_executable_properties)
      return true;

   return false;
}

'''

CACHE_OBJECT = r'''   /* Eden PS5: the object and the serialized shaders it holds live as long as the pipeline cache,
    * and are made on the threads that build pipelines; keep them out of those threads' heap
    * arenas. Only with the default allocator: radv_pipeline_cache_object_destroy() frees through
    * the device allocator, whose pfnFree is the process free only then (the emulator's heap owns
    * the block by address, whatever allocated it). */
   struct radv_pipeline_cache_object *object;
   if (eden_heap_long_lived_calloc && device->alloc.pfnFree == vk_default_allocator()->pfnFree)
      object = eden_heap_long_lived_calloc(1, size);
   else
      object = vk_alloc(&device->alloc, size, 8, VK_SYSTEM_ALLOCATION_SCOPE_CACHE);
'''

DESCRIPTOR_LAYOUT_OLD = r'''   /* Because we're reference counting and lifetimes may not be what the
    * client expects, these have to be allocated off the device and not as
    * their own object.
    */
   struct vk_descriptor_set_layout *layout =
      vk_zalloc(&device->alloc, size, 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
'''
DESCRIPTOR_LAYOUT_NEW = r'''   /* Because we're reference counting and lifetimes may not be what the
    * client expects, these have to be allocated off the device and not as
    * their own object.
    *
    * Eden PS5: a descriptor set layout lives as long as the pipelines that reference it and is
    * created on the threads that build those pipelines, so it is allocated from the emulator's
    * shared long-lived arena set: a pipeline object in a compile thread's heap arena would pin
    * that arena for the life of the pipeline. The matching free is
    * vk_descriptor_set_layout_destroy() -> vk_object_free() -> the device allocator's pfnFree,
    * which is the process free (which the emulator's heap owns by address) only with the default
    * allocator; an application-provided allocator keeps vk_zalloc.
    */
   struct vk_descriptor_set_layout *layout;
   if (eden_heap_long_lived_calloc && device->alloc.pfnFree == vk_default_allocator()->pfnFree)
      layout = eden_heap_long_lived_calloc(1, size);
   else
      layout = vk_zalloc(&device->alloc, size, 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
'''

# Each file is recognised by the code this script leaves in it, not by the exact blocks below:
# patch-radv-pipeline-oom.py refines some of them further (NULL checks inside the allocations).
EDITS = {
    'src/amd/vulkan/radv_shader.c': {
        'markers': ['#include "vk_eden_heap.h"', 'bool\nradv_keep_binary_debug_info(',
                    'radv_ps5_long_lived_calloc'],
        'edits': [
        ('#include "vk_debug_report.h"\n#include "vk_nir.h"\n',
         '#include "vk_debug_report.h"\n#include "vk_eden_heap.h"\n#include "vk_nir.h"\n'),
        ('''   return compiler_info->debug.dump_shader_stats && !is_meta_shader(nir);
}

void
radv_optimize_nir''',
         '   return compiler_info->debug.dump_shader_stats && !is_meta_shader(nir);\n}\n\n' +
         KEEP_DEBUG_INFO + 'void\nradv_optimize_nir'),
        ('   shader->code = calloc(shader->code_size, 1);\n',
         '   shader->code = radv_ps5_long_lived_calloc(shader->code_size, 1);\n'),
        ('''      dbg->ir_string = bin->llvm_ir_size ? strdup((const char *)(bin->data + bin->elf_size)) : NULL;
      dbg->disasm_string = malloc(disasm_size + 1);
      memcpy(dbg->disasm_string, disasm_data, disasm_size);
''',
         '''      dbg->ir_string =
         bin->llvm_ir_size ? radv_ps5_long_lived_strdup((const char *)(bin->data + bin->elf_size)) : NULL;
      dbg->disasm_string = radv_ps5_long_lived_malloc(disasm_size + 1);
      memcpy(dbg->disasm_string, disasm_data, disasm_size);
'''),
        ('''      if (bin->stats_size) {
         dbg->statistics = calloc(bin->stats_size, 1);
         memcpy(dbg->statistics, layout.stats, bin->stats_size);
      }

      dbg->ir_string = bin->ir_size ? strdup(layout.ir) : NULL;
      dbg->disasm_string = bin->disasm_size ? strdup(layout.disasm) : NULL;

      if (bin->debug_info_size) {
         dbg->debug_info = malloc(bin->debug_info_size);
         memcpy(dbg->debug_info, layout.debug_info, bin->debug_info_size);
''',
         '''      if (bin->stats_size) {
         dbg->statistics = radv_ps5_long_lived_calloc(bin->stats_size, 1);
         memcpy(dbg->statistics, layout.stats, bin->stats_size);
      }

      dbg->ir_string = bin->ir_size ? radv_ps5_long_lived_strdup(layout.ir) : NULL;
      dbg->disasm_string = bin->disasm_size ? radv_ps5_long_lived_strdup(layout.disasm) : NULL;

      if (bin->debug_info_size) {
         dbg->debug_info = radv_ps5_long_lived_malloc(bin->debug_info_size);
         memcpy(dbg->debug_info, layout.debug_info, bin->debug_info_size);
'''),
        ('   struct radv_shader *shader = calloc(1, sizeof(struct radv_shader));\n',
         '   struct radv_shader *shader = radv_ps5_long_lived_calloc(1, sizeof(struct radv_shader));\n'),
        ('   shader_part = calloc(1, sizeof(struct radv_shader_part));\n',
         '   shader_part = radv_ps5_long_lived_calloc(1, sizeof(struct radv_shader_part));\n'),
        ('   shader_part->disasm_string = binary->disasm_size ? strdup((const char *)(binary->data + binary->code_size)) : NULL;\n',
         '   shader_part->disasm_string =\n'
         '      binary->disasm_size ? radv_ps5_long_lived_strdup((const char *)(binary->data + binary->code_size)) : NULL;\n'),
        ('''   struct radv_shader *shader;
   radv_shader_create_uncached(device, binary, false, NULL, &debug, &shader);
   radv_parse_binary_debug_info(&device->compiler_info, binary, &shader->dbg);
''',
         '''   struct radv_shader *shader;
   radv_shader_create_uncached(device, binary, false, NULL, &debug, &shader);
   if (dump_shader)
      radv_parse_binary_debug_info(&device->compiler_info, binary, &shader->dbg);
'''),
        ('   if (!prolog || radv_parse_binary_debug_info(compiler_info, binary, &prolog->dbg) != VK_SUCCESS)\n',
         '   if (!prolog || (radv_keep_binary_debug_info(device) &&\n'
         '                   radv_parse_binary_debug_info(compiler_info, binary, &prolog->dbg) != VK_SUCCESS))\n'),
    ],
    },
    'src/amd/vulkan/radv_shader.h': {
        'markers': ['bool radv_keep_binary_debug_info(const struct radv_device *device);'],
        'edits': [
        ('''struct radv_shader_args;

VkResult radv_parse_binary_debug_info''',
         '''struct radv_shader_args;

bool radv_keep_binary_debug_info(const struct radv_device *device);

VkResult radv_parse_binary_debug_info'''),
    ],
    },
    'src/amd/vulkan/radv_pipeline_cache.c': {
        'markers': ['#include "vk_eden_heap.h"', 'radv_keep_binary_debug_info(device)',
                    'eden_heap_long_lived_calloc'],
        'edits': [
        ('#include "vk_alloc.h"\n#include "vk_pipeline.h"\n',
         '#include "vk_alloc.h"\n#include "vk_eden_heap.h"\n#include "vk_pipeline.h"\n'),
        ('''   radv_parse_binary_debug_info(&device->compiler_info, binary, &shader->dbg);

   assert(key_size == sizeof(shader->hash));
''',
         '''   /* Eden PS5: only when a debug option will read it (see radv_keep_binary_debug_info). */
   if (radv_keep_binary_debug_info(device))
      radv_parse_binary_debug_info(&device->compiler_info, binary, &shader->dbg);

   assert(key_size == sizeof(shader->hash));
'''),
        ('''      shader->dbg.spirv = malloc(shader->dbg.spirv_size);
      blob_copy_bytes(blob, shader->dbg.spirv, shader->dbg.spirv_size);

      const char *nir_string = blob_read_string(blob);
      const char *args_string = blob_read_string(blob);
      shader->dbg.nir_string = (nir_string && nir_string[0]) ? strdup(nir_string) : NULL;
      shader->dbg.args_string = (args_string && args_string[0]) ? strdup(args_string) : NULL;
''',
         '''      shader->dbg.spirv = radv_ps5_long_lived_malloc(shader->dbg.spirv_size);
      blob_copy_bytes(blob, shader->dbg.spirv, shader->dbg.spirv_size);

      const char *nir_string = blob_read_string(blob);
      const char *args_string = blob_read_string(blob);
      shader->dbg.nir_string = (nir_string && nir_string[0]) ? radv_ps5_long_lived_strdup(nir_string) : NULL;
      shader->dbg.args_string = (args_string && args_string[0]) ? radv_ps5_long_lived_strdup(args_string) : NULL;
'''),
        ('''      /* radv_parse_binary_debug_info() might have been done earlier by radv_shader_dump_asm(). Skip it in that case to
       * avoid memory leaks. */
      if (!dbg || (!dbg->statistics && !dbg->ir_string && !dbg->disasm_string && !dbg->debug_info_count))
         radv_parse_binary_debug_info(&device->compiler_info, binary, &shader->dbg);
''',
         '''      /* radv_parse_binary_debug_info() might have been done earlier by radv_shader_dump_asm(). Skip it in that case to
       * avoid memory leaks. Eden PS5: and skip it when no debug option will read it at all
       * (radv_keep_binary_debug_info). */
      if (radv_keep_binary_debug_info(device) &&
          (!dbg || (!dbg->statistics && !dbg->ir_string && !dbg->disasm_string && !dbg->debug_info_count)))
         radv_parse_binary_debug_info(&device->compiler_info, binary, &shader->dbg);
'''),
        ('   struct radv_pipeline_cache_object *object = vk_alloc(&device->alloc, size, 8, VK_SYSTEM_ALLOCATION_SCOPE_CACHE);\n',
         CACHE_OBJECT),
    ],
    },
    'src/vulkan/runtime/vk_descriptor_set_layout.c': {
        'markers': ['#include "vk_eden_heap.h"', 'eden_heap_long_lived_calloc'],
        'edits': [
        ('#include "vk_device.h"\n\nstatic void',
         '#include "vk_device.h"\n#include "vk_eden_heap.h"\n\nstatic void'),
        (DESCRIPTOR_LAYOUT_OLD, DESCRIPTOR_LAYOUT_NEW),
    ],
    },
}


def apply(source):
    changed = False
    header = source / 'src/vulkan/util/vk_eden_heap.h'
    if not header.is_file() or header.read_text() != HEADER:
        header.write_text(HEADER)
        changed = True
    for relative, spec in EDITS.items():
        path = source / relative
        text = path.read_text()
        present = [marker in text for marker in spec['markers']]
        if all(present):
            continue
        if any(present):
            raise SystemExit(f'{relative}: partially applied long-lived heap patch')
        for old, new in spec['edits']:
            if text.count(old) != 1:
                raise SystemExit(f'{relative}: pinned anchor changed ({text.count(old)}x): {old.splitlines()[0]!r}')
            text = text.replace(old, new)
        path.write_text(text)
        changed = True
    return changed


if __name__ == '__main__':
    source = (Path(sys.argv[1]) if len(sys.argv) > 1 else
              root.parent / 'mihawk-vulkan-review/.deps/work/radv-src')
    changed = apply(source)
    if changed:
        # Upstream skips the whole build while its receipt names the pinned revision: set the
        # receipt aside so the next build compiles the changed files (Ninja keeps every other
        # object).
        receipt = root.parent / 'mihawk-vulkan-review/.deps/native/radv-release/PROVENANCE.txt'
        if source.resolve() == (root.parent / 'mihawk-vulkan-review/.deps/work/radv-src').resolve() and receipt.exists():
            receipt.replace(receipt.with_suffix('.base.txt'))
    print(f'RADV long-lived heap: pipeline-lifetime objects from Eden\'s shared arenas, changed={changed}')
