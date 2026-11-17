#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pipeline creation fails cleanly when a shader allocation is refused, not at NULL + 0x10a.

On a full pool radv_shader_create() returns NULL and the pipeline create path still reported
VK_SUCCESS and handed the NULL shader to radv_compute_pipeline_init(), which dereferences it
through radv_get_user_sgpr_info() (fault at address 0x10a). A compute, graphics or ray-tracing
pipeline now reports VK_ERROR_OUT_OF_HOST_MEMORY instead, never caches a pipeline that lost a
shader, and drops a cache entry (in memory or on disk) that lacks a stage the pipeline needs or,
for graphics, the synthesized noop fragment shader. The shader upload/deserialize paths check
their allocations, and a failed shader is destroyed once through radv_shader_unref() rather than
freed twice or leaked. Applied after patch-radv-longlived.py: the checks sit on the allocations
it introduces. Patches the build export, never the Mesa checkout.
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

COMPUTE_COMPILE_OLD = r'''   struct radv_shader_debug_info cs_dbg = {0};
   struct radv_shader_binary *cs_binary =
      radv_compile_cs(compiler_info, &cs_stage, pipeline->base.is_internal, &cs_dbg);
   pipeline->base.shaders[MESA_SHADER_COMPUTE] =
      radv_shader_create(device, cache, cs_binary, skip_shaders_cache, &cs_dbg);

   cs_stage.feedback.duration += os_time_get_nano() - stage_start;

   if (!skip_shaders_cache) {
      radv_pipeline_cache_insert(device, cache, &pipeline->base);
   }

   free(cs_binary);
   if (radv_can_dump_shader_stats(&device->compiler_info, cs_stage.nir)) {
      radv_dump_shader_stats(device, &pipeline->base, pipeline->base.shaders[MESA_SHADER_COMPUTE], stderr);
   }
   radv_pipeline_stage_finish(&cs_stage);
'''
COMPUTE_COMPILE_NEW = r'''   struct radv_shader_debug_info cs_dbg = {0};
   struct radv_shader_binary *cs_binary =
      radv_compile_cs(compiler_info, &cs_stage, pipeline->base.is_internal, &cs_dbg);
   if (!cs_binary) {
      /* Eden PS5: codegen produced no binary (radv_postprocess_binary_config refused it); there
       * is nothing to create a shader from and nothing to free. */
      radv_pipeline_stage_finish(&cs_stage);
      result = VK_ERROR_UNKNOWN;
      goto done;
   }

   pipeline->base.shaders[MESA_SHADER_COMPUTE] =
      radv_shader_create(device, cache, cs_binary, skip_shaders_cache, &cs_dbg);

   cs_stage.feedback.duration += os_time_get_nano() - stage_start;

   if (!pipeline->base.shaders[MESA_SHADER_COMPUTE]) {
      /* Eden PS5: radv_shader_create() returns NULL when one of its allocations fails, which is
       * reachable when the emulator's shared pool is full. Do not cache a pipeline object with
       * no shader, and do not hand the NULL shader to radv_compute_pipeline_init(), which
       * dereferences it through radv_get_user_sgpr_info(); report the failure so the application
       * skips this pipeline and retries it later. */
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
   } else if (!skip_shaders_cache) {
      radv_pipeline_cache_insert(device, cache, &pipeline->base);
   }

   free(cs_binary);
   if (pipeline->base.shaders[MESA_SHADER_COMPUTE] &&
       radv_can_dump_shader_stats(&device->compiler_info, cs_stage.nir)) {
      radv_dump_shader_stats(device, &pipeline->base, pipeline->base.shaders[MESA_SHADER_COMPUTE], stderr);
   }
   radv_pipeline_stage_finish(&cs_stage);
'''

GRAPHICS_CACHE_OLD = r'''   bool found_in_application_cache = true;
   if (!skip_shaders_cache &&
       radv_graphics_pipeline_cache_search(device, cache, pipeline, &found_in_application_cache)) {
      if (found_in_application_cache)
         pipeline_feedback.flags |= VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT;

      if (retain_shaders) {
         /* For graphics pipeline libraries created with the RETAIN_LINK_TIME_OPTIMIZATION flag, we
          * need to retain the stage info because we can't know if the LTO pipelines will
          * be find in the shaders cache.
          */
         struct radv_graphics_lib_pipeline *gfx_pipeline_lib = radv_pipeline_to_graphics_lib(&pipeline->base);

         gfx_pipeline_lib->stages = radv_copy_shader_stage_create_info(device, pCreateInfo->stageCount,
                                                                       pCreateInfo->pStages, gfx_pipeline_lib->mem_ctx);
         if (!gfx_pipeline_lib->stages)
            return VK_ERROR_OUT_OF_HOST_MEMORY;

         gfx_pipeline_lib->stage_count = pCreateInfo->stageCount;

         for (unsigned i = 0; i < pCreateInfo->stageCount; i++) {
            mesa_shader_stage s = vk_to_mesa_shader_stage(pCreateInfo->pStages[i].stage);
            gfx_pipeline_lib->stage_keys[s] = gfx_state->key.stage_info[s];
         }
      }

      result = VK_SUCCESS;
      goto done;
   }
'''
GRAPHICS_CACHE_NEW = r'''   bool found_in_application_cache = true;
   if (!skip_shaders_cache &&
       radv_graphics_pipeline_cache_search(device, cache, pipeline, &found_in_application_cache)) {
      /* Eden PS5: an entry written by a build whose shader allocation failed can lack a stage
       * this pipeline needs (Eden's shared pool was full). Drop the entry and compile the
       * pipeline again instead of initializing and using it without that stage. */
      bool cache_entry_complete = true;
      for (unsigned i = 0; i < MESA_VULKAN_SHADER_STAGES; i++) {
         if (stages[i].stage != MESA_SHADER_NONE && !pipeline->base.shaders[i]) {
            cache_entry_complete = false;
            break;
         }
      }
      if (radv_pipeline_needs_noop_fs(pipeline, &gfx_state->key.gfx_state) &&
          !pipeline->base.shaders[MESA_SHADER_FRAGMENT])
         cache_entry_complete = false;

      if (!cache_entry_complete) {
         for (unsigned i = 0; i < MESA_VULKAN_SHADER_STAGES; i++) {
            if (pipeline->base.shaders[i]) {
               radv_shader_unref(device, pipeline->base.shaders[i]);
               pipeline->base.shaders[i] = NULL;
            }
         }
         if (pipeline->base.gs_copy_shader) {
            radv_shader_unref(device, pipeline->base.gs_copy_shader);
            pipeline->base.gs_copy_shader = NULL;
         }
         vk_pipeline_cache_object_unref(&device->vk, pipeline->base.cache_object);
         pipeline->base.cache_object = NULL;
      } else {
         if (found_in_application_cache)
            pipeline_feedback.flags |= VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT;

         if (retain_shaders) {
            /* For graphics pipeline libraries created with the RETAIN_LINK_TIME_OPTIMIZATION flag, we
             * need to retain the stage info because we can't know if the LTO pipelines will
             * be find in the shaders cache.
             */
            struct radv_graphics_lib_pipeline *gfx_pipeline_lib = radv_pipeline_to_graphics_lib(&pipeline->base);

            gfx_pipeline_lib->stages = radv_copy_shader_stage_create_info(device, pCreateInfo->stageCount,
                                                                          pCreateInfo->pStages, gfx_pipeline_lib->mem_ctx);
            if (!gfx_pipeline_lib->stages)
               return VK_ERROR_OUT_OF_HOST_MEMORY;

            gfx_pipeline_lib->stage_count = pCreateInfo->stageCount;

            for (unsigned i = 0; i < pCreateInfo->stageCount; i++) {
               mesa_shader_stage s = vk_to_mesa_shader_stage(pCreateInfo->pStages[i].stage);
               gfx_pipeline_lib->stage_keys[s] = gfx_state->key.stage_info[s];
            }
         }

         result = VK_SUCCESS;
         goto done;
      }
   }
'''

EDITS = {
    'src/amd/vulkan/radv_pipeline_compute.c': [
        (COMPUTE_COMPILE_OLD, COMPUTE_COMPILE_NEW),
        ('''   if (result != VK_SUCCESS) {
      radv_pipeline_destroy(device, &pipeline->base, pAllocator);
      return result;
   }

   radv_compute_pipeline_init(pipeline, pipeline_layout, pipeline->base.shaders[MESA_SHADER_COMPUTE]);
''',
         '''   if (result != VK_SUCCESS) {
      radv_pipeline_destroy(device, &pipeline->base, pAllocator);
      return result;
   }

   /* Eden PS5: radv_compute_pipeline_init() dereferences the shader through
    * radv_get_user_sgpr_info(). A pipeline without one (a failure path that did not report
    * itself, or a shader-less object from a cache) must fail instead of faulting at 0x10a. */
   if (!pipeline->base.shaders[MESA_SHADER_COMPUTE]) {
      radv_pipeline_destroy(device, &pipeline->base, pAllocator);
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   radv_compute_pipeline_init(pipeline, pipeline_layout, pipeline->base.shaders[MESA_SHADER_COMPUTE]);
'''),
    ],
    'src/amd/vulkan/radv_pipeline_cache.c': [
        ('''   assert(pipeline_obj->num_shaders == 1);
   pipeline->base.shaders[MESA_SHADER_COMPUTE] = radv_shader_ref(pipeline_obj->shaders[0]);
''',
         '''   if (pipeline_obj->num_shaders != 1) {
      /* Eden PS5: an out-of-memory build used to cache a pipeline object with no shader under
       * this key, and a disk cache can still hold one. Ignore it and compile the pipeline again
       * rather than reading past the object's shader array. */
      vk_pipeline_cache_object_unref(&device->vk, &pipeline_obj->base);
      return false;
   }
   pipeline->base.shaders[MESA_SHADER_COMPUTE] = radv_shader_ref(pipeline_obj->shaders[0]);
'''),
        ('''   pipeline_obj = radv_pipeline_cache_object_search(device, cache, &pipeline->base, found_in_application_cache);
   if (!pipeline_obj)
      return false;

   for (unsigned i = 0; i < pipeline_obj->num_shaders; i++) {
''',
         '''   pipeline_obj = radv_pipeline_cache_object_search(device, cache, &pipeline->base, found_in_application_cache);
   if (!pipeline_obj)
      return false;

   if (pipeline_obj->num_shaders == 0 &&
       !(pipeline->base.create_flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR)) {
      /* Eden PS5: a complete graphics pipeline always has a shader. An object with none is what
       * a failed build left behind (or a disk cache written by one); compile it again. */
      vk_pipeline_cache_object_unref(&device->vk, &pipeline_obj->base);
      return false;
   }

   for (unsigned i = 0; i < pipeline_obj->num_shaders; i++) {
'''),
        ('''      struct radv_shader *shader;
      radv_shader_create_uncached(device, binary, false, NULL, dbg, &shader);
      /* radv_parse_binary_debug_info() might have been done earlier by radv_shader_dump_asm(). Skip it in that case to
''',
         '''      struct radv_shader *shader;
      radv_shader_create_uncached(device, binary, false, NULL, dbg, &shader);
      if (!shader)
         return NULL;
      /* radv_parse_binary_debug_info() might have been done earlier by radv_shader_dump_asm(). Skip it in that case to
'''),
        ('''      shader->dbg.spirv = radv_ps5_long_lived_malloc(shader->dbg.spirv_size);
      blob_copy_bytes(blob, shader->dbg.spirv, shader->dbg.spirv_size);
''',
         '''      shader->dbg.spirv = radv_ps5_long_lived_malloc(shader->dbg.spirv_size);
      if (shader->dbg.spirv_size && !shader->dbg.spirv) {
         /* Eden PS5: the pool is full; a cache miss and a compile are better than a copy to
          * NULL. */
         radv_shader_unref(device, shader);
         return NULL;
      }
      blob_copy_bytes(blob, shader->dbg.spirv, shader->dbg.spirv_size);
'''),
    ],
    'src/amd/vulkan/radv_shader.c': [
        ('''   shader->code = radv_ps5_long_lived_calloc(shader->code_size, 1);
   if (!shader->code) {
      radv_shader_unref(device, shader);
      return false;
   }
''',
         '''   shader->code = radv_ps5_long_lived_calloc(shader->code_size, 1);
   if (!shader->code) {
      /* Eden PS5: the caller (radv_shader_create_uncached) destroys the shader once on any
       * failure; dropping the reference here freed it twice. */
      return false;
   }
'''),
        ('''      if (!radv_open_rtld_binary(pdev->info.gfx_level, binary, &rtld_binary)) {
         free(shader);
         return false;
      }
''',
         '''      if (!radv_open_rtld_binary(pdev->info.gfx_level, binary, &rtld_binary)) {
         /* Eden PS5: freed once by the caller, together with shader->code and the GPU
          * allocation. */
         return false;
      }
'''),
        ('''      if (!ac_rtld_upload(&info)) {
         radv_shader_unref(device, shader);
         ac_rtld_close(&rtld_binary);
         return false;
      }
''',
         '''      if (!ac_rtld_upload(&info)) {
         /* Eden PS5: the caller destroys the shader once. */
         ac_rtld_close(&rtld_binary);
         return false;
      }
'''),
        ('''out:
   if (result != VK_SUCCESS) {
      free(shader);
      *out_shader = NULL;
   }
''',
         '''out:
   if (result != VK_SUCCESS) {
      /* Eden PS5: destroy the partial shader through its destructor. free() alone leaked
       * shader->code and the shader's GPU allocation, and a failure path in
       * radv_shader_binary_upload() had already dropped the last reference. */
      if (shader)
         radv_shader_unref(device, shader);
      *out_shader = NULL;
   }
'''),
        ('''   struct radv_shader *shader;
   radv_shader_create_uncached(device, binary, false, NULL, &debug, &shader);
   if (dump_shader)
      radv_parse_binary_debug_info(&device->compiler_info, binary, &shader->dbg);
''',
         '''   struct radv_shader *shader;
   radv_shader_create_uncached(device, binary, false, NULL, &debug, &shader);
   if (!shader) {
      /* Eden PS5: no allocation for the trap handler shader; the caller runs without one. */
      ralloc_free(b.shader);
      free(binary);
      return NULL;
   }
   if (dump_shader)
      radv_parse_binary_debug_info(&device->compiler_info, binary, &shader->dbg);
'''),
        ('''      dbg->disasm_string = radv_ps5_long_lived_malloc(disasm_size + 1);
      memcpy(dbg->disasm_string, disasm_data, disasm_size);
''',
         '''      dbg->disasm_string = radv_ps5_long_lived_malloc(disasm_size + 1);
      if (!dbg->disasm_string) {
         ac_rtld_close(&rtld_binary);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
      memcpy(dbg->disasm_string, disasm_data, disasm_size);
'''),
        ('''      if (bin->stats_size) {
         dbg->statistics = radv_ps5_long_lived_calloc(bin->stats_size, 1);
         memcpy(dbg->statistics, layout.stats, bin->stats_size);
      }
''',
         '''      if (bin->stats_size) {
         dbg->statistics = radv_ps5_long_lived_calloc(bin->stats_size, 1);
         if (!dbg->statistics)
            return VK_ERROR_OUT_OF_HOST_MEMORY;
         memcpy(dbg->statistics, layout.stats, bin->stats_size);
      }
'''),
        ('''      if (bin->debug_info_size) {
         dbg->debug_info = radv_ps5_long_lived_malloc(bin->debug_info_size);
         memcpy(dbg->debug_info, layout.debug_info, bin->debug_info_size);
''',
         '''      if (bin->debug_info_size) {
         dbg->debug_info = radv_ps5_long_lived_malloc(bin->debug_info_size);
         if (!dbg->debug_info)
            return VK_ERROR_OUT_OF_HOST_MEMORY;
         memcpy(dbg->debug_info, layout.debug_info, bin->debug_info_size);
'''),
    ],
    'src/amd/vulkan/radv_pipeline_graphics.c': [
        (GRAPHICS_CACHE_OLD, GRAPHICS_CACHE_NEW),
        ('''   radv_graphics_shaders_create(device, cache, skip_shaders_cache, pipeline->base.shaders, binaries, debug,
                                &pipeline->base.gs_copy_shader, gs_copy_binary, &gs_copy_debug);
''',
         '''   radv_graphics_shaders_create(device, cache, skip_shaders_cache, pipeline->base.shaders, binaries, debug,
                                &pipeline->base.gs_copy_shader, gs_copy_binary, &gs_copy_debug);

   /* Eden PS5: radv_shader_create() returns NULL when an allocation fails (the emulator's shared
    * pool can be full). A pipeline missing one of its stages would be initialized and cached
    * without it; report the failure so the application skips this pipeline and retries it
    * later. */
   for (unsigned i = 0; i < MESA_VULKAN_SHADER_STAGES; i++) {
      if (binaries[i] && !pipeline->base.shaders[i])
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   if (gs_copy_binary && !pipeline->base.gs_copy_shader)
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
'''),
        ('''      for (unsigned i = 0; i < RADV_GS_COMPUTE_SHADERS; i++) {
         if (gs_compute.binaries[i] && pipeline->base.gs_compute)
            pipeline->base.gs_compute->shaders[i] =
               radv_shader_create(device, cache, gs_compute.binaries[i], true, &gs_compute.debug[i]);
         free(gs_compute.binaries[i]);
      }
''',
         '''      for (unsigned i = 0; i < RADV_GS_COMPUTE_SHADERS; i++) {
         if (gs_compute.binaries[i] && pipeline->base.gs_compute) {
            pipeline->base.gs_compute->shaders[i] =
               radv_shader_create(device, cache, gs_compute.binaries[i], true, &gs_compute.debug[i]);
            /* Eden PS5: same allocation failure as above. */
            if (!pipeline->base.gs_compute->shaders[i])
               result = VK_ERROR_OUT_OF_HOST_MEMORY;
         }
         free(gs_compute.binaries[i]);
      }
'''),
        ('''   if (!skip_shaders_cache) {
      radv_pipeline_cache_insert(device, cache, &pipeline->base);
   }
''',
         '''   /* Eden PS5: never cache a pipeline that lost a shader. */
   if (!skip_shaders_cache && result == VK_SUCCESS) {
      radv_pipeline_cache_insert(device, cache, &pipeline->base);
   }
'''),
    ],
    'src/amd/vulkan/radv_pipeline_rt.c': [
        ('''      compute_rt_stack_size(pCreateInfo, pipeline);
      compile_rt_prolog(device, pipeline);

      radv_compute_pipeline_init(&pipeline->base, pipeline_layout, pipeline->prolog);
''',
         '''      compute_rt_stack_size(pCreateInfo, pipeline);
      compile_rt_prolog(device, pipeline);
      if (!pipeline->prolog) {
         /* Eden PS5: the prolog shader could not be allocated and radv_compute_pipeline_init()
          * dereferences it. */
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
         goto fail;
      }

      radv_compute_pipeline_init(&pipeline->base, pipeline_layout, pipeline->prolog);
'''),
    ],
}


def apply(source):
    changed = False
    for relative, replacements in EDITS.items():
        path = source / relative
        text = path.read_text()
        applied = [new in text for old, new in replacements]
        if all(applied):
            continue
        if any(applied):
            raise SystemExit(f'{relative}: partially applied pipeline-OOM patch')
        for old, new in replacements:
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
    print(f'RADV pipeline OOM: failed shader allocations fail the pipeline; changed={changed}')
