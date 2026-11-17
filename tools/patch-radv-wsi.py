#!/usr/bin/env python3
"""Apply the pinned Eden display-lifetime adaptation to the Mesa build export."""
from pathlib import Path
import hashlib
import sys

root = Path(__file__).resolve().parents[1]
source = root.parent/'mihawk-mesa-review/src/vulkan/wsi/wsi_common_videoout.c'
target = root.parent/'mihawk-vulkan-review/.deps/work/radv-src/src/vulkan/wsi/wsi_common_videoout.c'
text = source.read_text()

def replace(old, new):
    global text
    assert text.count(old) == 1, f'Pinned WSI anchor changed: {old[:70]}'
    text = text.replace(old, new)

replace('   bool opened;\n', '   unsigned owners;\n   bool closing, stop;\n   bool opened;\n')
# A set registers the number of buffers its swapchain uses, not the five a
# swapchain may ask for: Eden asks for three, so two 32 MiB framebuffers a size
# stay in the pool the guest shares (64 MiB at 4K, 20 MiB at 1080p).
replace('/* Buffers in each size\'s set. */\n#define VIDEOOUT_BUFFERS 5',
        '/* The most buffers a swapchain may ask for; a set registers the count its own\n'
        ' * swapchain uses (Eden asks for three), so VideoOut holds only those. */\n#define VIDEOOUT_BUFFERS 5')
replace(''' * names the buffer of whichever set it is in. Three sets use buffers 0-14:
 * VideoOut refused a fourth set's buffers 15-19 (0x80290001), and presents
 * after that refusal flickered, so no size is added without making room. A swapchain takes the first of''',
        ''' * names the buffer of whichever set it is in. A set registers only the
 * buffers its swapchain uses and the next set starts after them, so the three
 * sizes use buffers 0-8: VideoOut refused a fourth set at buffers 15-19
 * (0x80290001), and presents after that refusal flickered, so no size is added
 * without making room. A swapchain takes the first of''')
# A set of framebuffers per output size: each remembers the memory to give back,
# and the buffer indices it was registered as.
replace('''   /* A set of buffers for each size a swapchain has used, in the order they
    * were registered; set i's buffers are indices i * VIDEOOUT_BUFFERS on. */''',
        '''   /* A set of buffers for each size a swapchain has used, in the order they
    * were registered; a set's buffers are the indices first .. first + count - 1. */''')
replace('      uint8_t *buffers;\n   } sets[VIDEOOUT_SIZES];',
        '      uint8_t *buffers;\n      int64_t physical;\n      size_t bytes;\n'
        '      /* The buffer indices the set\'s count buffers are registered as. */\n'
        '      uint32_t first, count;\n   } sets[VIDEOOUT_SIZES];')
replace('   uint32_t set_count;\n',
        '   uint32_t set_count;\n   /* The first VideoOut buffer index no set uses. */\n   uint32_t next_buffer;\n')
replace('static int\nsceVideoOutSetFlipRate', 'static int\nsceVideoOutClose(int32_t handle)\n{\n   return 0;\n}\n\nstatic int\nsceVideoOutSetFlipRate')
replace('   uint8_t *buffers = NULL;\n#if defined(__PROSPERO__)\n   int64_t physical = -1;',
        '   uint8_t *buffers = NULL;\n   int64_t physical = -1;\n#if defined(__PROSPERO__)')
# A set is registered with the number of buffers its swapchain uses; a later
# swapchain of the same size that wants more is refused rather than handed
# indices VideoOut does not know.
replace('''static VkResult
videoout_register_locked(const struct wsi_videoout_size *size, const struct wsi_videoout_set **set_out)
{
   struct wsi_videoout_output *out = &videoout_output;
   for (uint32_t i = 0; i < out->set_count; i++) {
      if (out->sets[i].size == size) {
         *set_out = &out->sets[i];
         return VK_SUCCESS;
      }
   }
   if (!videoout_open_locked())
      return VK_ERROR_INITIALIZATION_FAILED;

   const uint32_t index = out->set_count;
   const size_t bytes = VIDEOOUT_BUFFERS * size->buffer_bytes;''',
        '''static VkResult
videoout_register_locked(const struct wsi_videoout_size *size, uint32_t buffer_count,
                         const struct wsi_videoout_set **set_out)
{
   struct wsi_videoout_output *out = &videoout_output;
   for (uint32_t i = 0; i < out->set_count; i++) {
      if (out->sets[i].size == size) {
         /* A set keeps the buffers of the count it was registered with. */
         if (buffer_count > out->sets[i].count) {
            fprintf(stderr, "wsi/videoout: the %ux%u set has %u buffers; a swapchain asking for %u is refused\\n",
                    size->extent.width, size->extent.height, out->sets[i].count, buffer_count);
            return VK_ERROR_INITIALIZATION_FAILED;
         }
         *set_out = &out->sets[i];
         return VK_SUCCESS;
      }
   }
   if (!videoout_open_locked())
      return VK_ERROR_INITIALIZATION_FAILED;
   if (out->next_buffer + buffer_count > ARRAY_SIZE(out->buffer)) {
      fprintf(stderr, "wsi/videoout: no VideoOut buffer index left for a set of %u\\n", buffer_count);
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   const uint32_t index = out->set_count;
   const uint32_t first = out->next_buffer;
   const size_t bytes = buffer_count * size->buffer_bytes;''')
replace('''   struct ps5_video_out_buffer registered[VIDEOOUT_BUFFERS];
   for (uint32_t i = 0; i < VIDEOOUT_BUFFERS; i++)''',
        '''   struct ps5_video_out_buffer registered[VIDEOOUT_BUFFERS];
   for (uint32_t i = 0; i < buffer_count; i++)''')
replace('''      result = sceVideoOutRegisterBuffers2(out->handle, (int32_t)index, (int32_t)(index * VIDEOOUT_BUFFERS),
                                           registered, VIDEOOUT_BUFFERS, attribute, 0, NULL);''',
        '''      result = sceVideoOutRegisterBuffers2(out->handle, (int32_t)index, (int32_t)first, registered,
                                           (int32_t)buffer_count, attribute, 0, NULL);''')
replace('''   fprintf(stderr, "wsi/videoout: %ux%u framebuffers registered as set %u, buffers %u-%u\\n", size->extent.width,
           size->extent.height, index, index * VIDEOOUT_BUFFERS, index * VIDEOOUT_BUFFERS + VIDEOOUT_BUFFERS - 1);
   out->sets[index] = (struct wsi_videoout_set){.size = size, .buffers = buffers};
   out->set_count = index + 1;''',
        '''   fprintf(stderr, "wsi/videoout: %ux%u framebuffers registered as set %u, buffers %u-%u\\n",
           size->extent.width, size->extent.height, index, first, first + buffer_count - 1);
   out->sets[index] = (struct wsi_videoout_set){.size = size, .buffers = buffers, .physical = physical,
                                                .bytes = bytes, .first = first, .count = buffer_count};
   out->set_count = index + 1;
   out->next_buffer = first + buffer_count;''')
replace('      while (out->queue_count == 0)\n         cnd_wait(&out->changed, &out->lock);',
        '      while (out->queue_count == 0 && !out->stop)\n         cnd_wait(&out->changed, &out->lock);\n      if (out->queue_count == 0 && out->stop) {\n         mtx_unlock(&out->lock);\n         return 0;\n      }')
replace('VkResult\nwsi_display_init_wsi',
        (root/'tools/radv-videoout-lifecycle.inc').read_text()+'\nVkResult\nwsi_display_init_wsi')
# 120 Hz: the app asks for it per game session (headless/display_refresh.h). The pinned source
# decides from the package's param.json, which the app cannot read once it has left the sandbox,
# and for the whole run; the package declares the capability (tools/package-headless-native.sh).
declares = 'videoout_title_declares_high_frame_rate(void)\n{\n'
assert text.count(declares) == 1, 'Pinned WSI anchor changed: the 120 Hz declaration'
body_start = text.index(declares) + len(declares)
body_end = text.index('\n}\n', body_start)
assert 'param.json' in text[body_start:body_end], 'Pinned WSI anchor changed: the 120 Hz declaration reads param.json'
text = (text[:body_start] + '   const char *const asked = getenv("EDEN_VIDEOOUT_120HZ");\n'
        '   return asked != NULL && asked[0] == \'1\';' + text[body_end:])
# The pinned source takes the 120 Hz output back unless a vblank then comes about every 8.3 ms.
# On a display with a variable refresh rate that is wrong: before the first frame is shown it
# idles at the bottom of its range (a vblank every 20.87 ms, 48 Hz, in two reports), the output
# went back to 59.94 Hz, and a game could not have 120 Hz at all. Only a display that stayed at
# 59.94 Hz (a vblank about every 16.7 ms) gives the mode back.
replace('         if (period > VIDEOOUT_HIGH_REFRESH_LIMIT_NS) {',
        '         if (period > UINT64_C(15000000) && period < UINT64_C(18500000)) {')
# Say so when the display follows, as the pinned source does when it does not.
replace('''                    (double)period / 1e6);
         }
      } else {''', '''                    (double)period / 1e6);
         } else {
            fprintf(stderr, "wsi/videoout: 119.88 Hz output, a vblank every %.3f ms\\n", (double)period / 1e6);
         }
      } else {''')
# The swapchain's image count is known before the set is registered, so the set
# gets exactly that many VideoOut buffers. chain->first is the set's own first
# buffer index, and the on-screen scan covers every registered slot.
replace('''   const struct wsi_videoout_set *set = NULL;
   mtx_lock(&out->lock);
   VkResult result = videoout_register_locked(mode->size, &set);
   if (result == VK_SUCCESS && out->current && out->current != (struct wsi_videoout_swapchain *)
                                                 wsi_swapchain_from_handle(pCreateInfo->oldSwapchain))
      result = VK_ERROR_NATIVE_WINDOW_IN_USE_KHR;
   mtx_unlock(&out->lock);
   if (result != VK_SUCCESS)
      return result;

   const uint32_t num_images = MAX2(pCreateInfo->minImageCount, 3);''',
        '''   /* The images the swapchain will have, and so the buffers the set registers: Eden asks
    * for minImageCount + 1 (three), and two of the five 32 MiB framebuffers a size stay in
    * the pool the guest shares. */
   const uint32_t num_images = MAX2(pCreateInfo->minImageCount, 3);
   const struct wsi_videoout_set *set = NULL;
   mtx_lock(&out->lock);
   VkResult result = videoout_register_locked(mode->size, num_images, &set);
   if (result == VK_SUCCESS && out->current && out->current != (struct wsi_videoout_swapchain *)
                                                 wsi_swapchain_from_handle(pCreateInfo->oldSwapchain))
      result = VK_ERROR_NATIVE_WINDOW_IN_USE_KHR;
   mtx_unlock(&out->lock);
   if (result != VK_SUCCESS)
      return result;''')
replace('   chain->first = (uint32_t)(set - out->sets) * VIDEOOUT_BUFFERS;', '   chain->first = set->first;')
replace('   for (uint32_t i = 0; i < out->set_count * VIDEOOUT_BUFFERS; i++)',
        '   for (uint32_t i = 0; i < ARRAY_SIZE(out->buffer); i++)')
replace('   wsi->base.get_support =', '   videoout_ref();\n\n   wsi->base.get_support =')
replace('   if (wsi)\n      vk_free(alloc, wsi);',
        '   if (wsi) {\n      videoout_unref();\n      vk_free(alloc, wsi);\n      wsi_device->wsi[VK_ICD_WSI_PLATFORM_DISPLAY] = NULL;\n   }')
if sys.argv[1:2] == ['--check']:
    # Is the driver that was built (its recorded display source, EDEN_WSI_SHA256) this adaptation?
    recorded = Path(sys.argv[2]).read_text().split()[0] if Path(sys.argv[2]).is_file() else ''
    sys.exit(0 if recorded == hashlib.sha256(text.encode()).hexdigest() else 1)
changed = not target.exists() or target.read_text() != text
if changed:
    target.write_text(text)
    # Upstream caches the whole build by pin. Invalidate only that receipt;
    # Ninja retains all unchanged objects and rebuilds the patched WSI.
    provenance = root.parent/'mihawk-vulkan-review/.deps/native/radv-release/PROVENANCE.txt'
    if provenance.exists():
        provenance.replace(provenance.with_suffix('.base.txt'))
print(f'RADV WSI adaptation changed={changed} sha256={hashlib.sha256(text.encode()).hexdigest()}')

# Preserve the reason behind RADV's successful-but-empty device enumeration.
# These startup-only diagnostics avoid repeated full game runs for each stage.
for relative, replacements in {
    # Restore the pinned upload path: explicit CPU flush did not affect HUD corruption.
    'src/amd/vulkan/radv_cmd_buffer.c': [],
    'src/amd/vulkan/radv_physical_device.c': [
        ('   if (result == VK_ERROR_INCOMPATIBLE_DRIVER || result == VK_ERROR_INITIALIZATION_FAILED)\n      return VK_SUCCESS;',
         '   if (result != VK_SUCCESS) fprintf(stderr, "radv/ps5: physical device creation result=%d\\n", result);\n   if (result == VK_ERROR_INCOMPATIBLE_DRIVER || result == VK_ERROR_INITIALIZATION_FAILED)\n      return VK_SUCCESS;'),
        ('   if (!pdev->addrlib) {', '   if (!pdev->addrlib) {\n      fprintf(stderr, "radv/ps5: addrlib creation failed\\n");'),
    ],
    'src/amd/vulkan/winsys/ps5/radv_ps5_winsys.c': [
        ('   if (!radv_ps5_platform_init())\n      return;', '   if (!radv_ps5_platform_init()) {\n      fprintf(stderr, "radv/ps5: queue platform initialization failed\\n");\n      return;\n   }'),
        ('   if (!radv_ps5_memory_alloc(RADV_PS5_RING_BYTES, RADV_PS5_LARGE_BYTES, true, &queue->ring))\n      return;', '   if (!radv_ps5_memory_alloc(RADV_PS5_RING_BYTES, RADV_PS5_LARGE_BYTES, true, &queue->ring)) {\n      fprintf(stderr, "radv/ps5: queue ring allocation failed\\n");\n      return;\n   }'),
        ('      radv_ps5_memory_free(&queue->ring);', '      fprintf(stderr, "radv/ps5: queue marker allocation failed\\n");\n      radv_ps5_memory_free(&queue->ring);'),
    ],
    'src/amd/vulkan/winsys/ps5/radv_ps5_platform.c': [
        ('''bool
radv_ps5_memory_grant_gpu(void *address, uint64_t bytes)
{
   /* A title's anonymous memory takes GPU access this way, and a shader then
    * writes it through the CPU's address (HARDWARE_FINDINGS.md, 2026-09-27). */
   return sceKernelMprotect(address, bytes,
                            PS5_KERNEL_PROT_CPU_READ | PS5_KERNEL_PROT_CPU_WRITE | PS5_KERNEL_PROT_GPU_READ |
                               PS5_KERNEL_PROT_GPU_WRITE) == 0;
}''', (root/'tools/radv-host-memory.inc').read_text().strip()),
        ('   if (!placed) {', '   if (!placed) {\n      fprintf(stderr, "radv/ps5: mapping rejected result=%d address=%p bytes=%llu window32=%d\\n", result, address, (unsigned long long)bytes, window32);'),
    ],
}.items():
    original = root.parent/'mihawk-mesa-review'/relative
    destination = root.parent/'mihawk-vulkan-review/.deps/work/radv-src'/relative
    adapted = original.read_text()
    for old, new in replacements:
        assert adapted.count(old) == 1, f'Pinned initialization anchor changed: {relative}'
        adapted = adapted.replace(old, new)
    if destination.read_text() != adapted:
        destination.write_text(adapted)
        provenance = root.parent/'mihawk-vulkan-review/.deps/native/radv-release/PROVENANCE.txt'
        if provenance.exists():
            provenance.replace(provenance.with_suffix('.base.txt'))
    print(f'RADV initialization diagnostic {relative} sha256={hashlib.sha256(adapted.encode()).hexdigest()}')
