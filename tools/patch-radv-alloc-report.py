#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Why a driver allocation was refused, on the console's shared pool (diagnostic).

A refused allocation ended a session with only the app's own line to show for it: on Zelda: Ocarina
of Time a 32 MiB staging buffer was refused while the pool's largest free run was 2 MiB, and the log
could not say whether the buffer asked for the 32-bit window (which patch-radv-pieces.py does not
piece) or the pool had too few 2 MiB runs left (2026-11-17 memory sweep). Every refusal of
radv_ps5_memory_alloc now prints the size and alignment asked for, whether it was a 32-bit window
buffer, the single run's result, how many of the 2 MiB pieces it needs the pieces path could take
and what stopped it, what the app gave back (eden_memory_recover), and the largest free run left.
The first eight refusals and then one a second, so a retrying app cannot fill the log. The memory
types the driver exposes, and which of them place a buffer in the 32-bit window, are printed once
at device creation, so an app's own memory-type lines (Eden's EDEN_MEM_SPLIT type=) can be read
against them. Runs after tools/patch-radv-pieces.py, whose allocation path it reports on. Patches
the build export, never the Mesa checkout.
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
source = (Path(sys.argv[1]) if len(sys.argv) > 1 else
          root.parent / 'mihawk-vulkan-review/.deps/work/radv-src')
MARK = 'EglantinePS5 allocation refusal report'

REPORT = r'''
/* ''' + MARK + r''' (tools/patch-radv-alloc-report.py): what a refused allocation asked for and
 * what the pool offered. The size and alignment, whether it was a 32-bit window buffer, the single
 * run's result, how many of the 2 MiB pieces it needs could be taken and what stopped them, what
 * the app gave back, and the largest free run left. */
struct radv_ps5_alloc_report {
   uint64_t bytes;
   uint64_t alignment;
   uint32_t pieces_wanted;
   uint32_t pieces_done;
   int32_t run_result;
   int32_t piece_result;
   int32_t map_result;
   size_t recovered;
   uint32_t rounds;
   bool window32;
   bool no_granule;
   bool moved;
};

static void
radv_ps5_alloc_refused(const struct radv_ps5_alloc_report *report)
{
   static uint32_t refusals;
   static int64_t last_ns;
   const uint32_t count = p_atomic_inc_return(&refusals);
   const int64_t now = os_time_get_nano();
   /* The first eight, then one a second: an allocation that keeps failing would fill the log. */
   if (count > 8) {
      const int64_t previous = p_atomic_read(&last_ns);
      if (now - previous < INT64_C(1000000000) || p_atomic_cmpxchg(&last_ns, previous, now) != previous)
         return;
   } else {
      p_atomic_set(&last_ns, now);
   }
   int64_t start = -1;
   size_t largest = 0;
   const int64_t pool = sceKernelGetDirectMemorySize();
   if (pool <= 0 || sceKernelAvailableDirectMemorySize(0, pool, RADV_PS5_PAGE_BYTES, &start, &largest) != 0)
      largest = 0;
   fprintf(stderr,
           "radv/ps5: alloc refused bytes=%" PRIu64 " align=%" PRIu64 " window32=%d run=0x%08x"
           " pieces=%u/%u%s piece=0x%08x map=0x%08x%s recovered=%zuMiB rounds=%u largest_free=%zuMiB"
           " (%u so far)\n",
           report->bytes, report->alignment, report->window32, (unsigned)report->run_result,
           report->pieces_done, report->pieces_wanted,
           report->no_granule ? " (no room for its address range)" : "", (unsigned)report->piece_result,
           (unsigned)report->map_result, report->moved ? " (mapped elsewhere)" : "",
           report->recovered >> 20, report->rounds, largest >> 20, count);
}
'''

edits = {
    'src/amd/vulkan/winsys/ps5/radv_ps5_platform.c': [(
        '''static bool
radv_ps5_memory_alloc_pieces(uint64_t bytes, uint64_t alignment, struct radv_ps5_memory *out)
{
''',
        REPORT + '''
static bool
radv_ps5_memory_alloc_pieces(uint64_t bytes, uint64_t alignment, struct radv_ps5_memory *out,
                             struct radv_ps5_alloc_report *report)
{
'''), (
        '''   if (granule == UINT32_MAX) {
      free(pieces);
      return false;
   }
''',
        '''   if (granule == UINT32_MAX) {
      report->no_granule = true;
      free(pieces);
      return false;
   }
'''), (
        '''   uint32_t done = 0;
   for (; done < granules; ++done) {
      int64_t physical = -1;
      if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), RADV_PS5_LARGE_BYTES, RADV_PS5_LARGE_BYTES,
                                        RADV_PS5_DIRECT_TYPE, &physical) != 0)
         break;
      void *const want = at + (uint64_t)done * RADV_PS5_LARGE_BYTES;
      void *address = want;
      const int32_t result =
         sceKernelMapDirectMemory(&address, RADV_PS5_LARGE_BYTES, RADV_PS5_PROTECTION, 0, physical, RADV_PS5_LARGE_BYTES);
      if (result != 0 || address != want) {
''',
        '''   uint32_t done = 0;
   for (; done < granules; ++done) {
      int64_t physical = -1;
      const int32_t take =
         sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), RADV_PS5_LARGE_BYTES, RADV_PS5_LARGE_BYTES,
                                       RADV_PS5_DIRECT_TYPE, &physical);
      if (take != 0) {
         report->piece_result = take;
         break;
      }
      void *const want = at + (uint64_t)done * RADV_PS5_LARGE_BYTES;
      void *address = want;
      const int32_t result =
         sceKernelMapDirectMemory(&address, RADV_PS5_LARGE_BYTES, RADV_PS5_PROTECTION, 0, physical, RADV_PS5_LARGE_BYTES);
      if (result != 0 || address != want) {
         report->map_result = result;
         report->moved = result == 0;
'''), (
        '''      pieces[done] = physical;
   }
   if (done != granules) {
''',
        '''      pieces[done] = physical;
   }
   report->pieces_done = done;
   if (done != granules) {
'''), (
        '''   int64_t physical = -1;
   if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), bytes, alignment,
                                     RADV_PS5_DIRECT_TYPE, &physical) != 0)
   {
      /* No run that long: pieces; else the app gives memory back (eden_memory_recover packs cold
       * guest memory) and the run, or pieces, are asked for once more. */
      if (!window32 && bytes >= RADV_PS5_LARGE_BYTES && radv_ps5_memory_alloc_pieces(bytes, alignment, out))
         return true;
      if (!eden_memory_recover || eden_memory_recover(bytes + (UINT64_C(16) << 20)) == 0)
         return false;
      if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), bytes, alignment,
                                        RADV_PS5_DIRECT_TYPE, &physical) != 0)
         return !window32 && bytes >= RADV_PS5_LARGE_BYTES && radv_ps5_memory_alloc_pieces(bytes, alignment, out);
   }
''',
        '''   /* ''' + MARK + r''': a refusal says what it asked for and what the pool offered. */
   int64_t physical = -1;
   int32_t run_result =
      sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), bytes, alignment,
                                    RADV_PS5_DIRECT_TYPE, &physical);
   if (run_result != 0)
   {
      struct radv_ps5_alloc_report report = {
         .bytes = bytes,
         .alignment = alignment,
         .window32 = window32,
         .run_result = run_result,
         .pieces_wanted = bytes >= RADV_PS5_LARGE_BYTES ? (uint32_t)DIV_ROUND_UP(bytes, RADV_PS5_LARGE_BYTES) : 0,
      };
      /* No run that long: pieces; else the app gives memory back (eden_memory_recover packs cold
       * guest memory) and the run, or pieces, are asked for once more. */
      if (!window32 && report.pieces_wanted &&
          radv_ps5_memory_alloc_pieces(bytes, alignment, out, &report))
         return true;
      if (eden_memory_recover &&
          (report.recovered = eden_memory_recover(bytes + (UINT64_C(16) << 20))) != 0) {
         report.rounds++;
         run_result = sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), bytes, alignment,
                                                   RADV_PS5_DIRECT_TYPE, &physical);
         report.run_result = run_result;
      }
      if (run_result != 0) {
         if (report.rounds && !window32 && report.pieces_wanted &&
             radv_ps5_memory_alloc_pieces(bytes, alignment, out, &report))
            return true;
         radv_ps5_alloc_refused(&report);
         return false;
      }
   }
''')],
    'src/amd/vulkan/radv_physical_device.c': [(
        '''   for (unsigned i = 0; i < type_count; ++i) {
      if (pdev->memory_flags[i] & RADEON_FLAG_32BIT)
         pdev->memory_types_32bit |= BITFIELD_BIT(i);
      if (pdev->memory_flags[i] & RADEON_FLAG_CPU_ACCESS)
         pdev->memory_types_host_visible |= BITFIELD_BIT(i);
      if (pdev->memory_flags[i] & RADEON_FLAG_ENCRYPTED)
         pdev->memory_types_protected |= BITFIELD_BIT(i);
   }
''',
        '''   for (unsigned i = 0; i < type_count; ++i) {
      if (pdev->memory_flags[i] & RADEON_FLAG_32BIT)
         pdev->memory_types_32bit |= BITFIELD_BIT(i);
      if (pdev->memory_flags[i] & RADEON_FLAG_CPU_ACCESS)
         pdev->memory_types_host_visible |= BITFIELD_BIT(i);
      if (pdev->memory_flags[i] & RADEON_FLAG_ENCRYPTED)
         pdev->memory_types_protected |= BITFIELD_BIT(i);
   }
   /* ''' + MARK + r''': the memory types exposed and which place a buffer in the 32-bit
    * window, once at device creation, so an app's own memory-type lines can be read against them. */
   for (unsigned i = 0; i < type_count; ++i)
      fprintf(stderr, "radv/ps5: memory type %u heap=%u flags=0x%x%s%s\n", i,
              (unsigned)pdev->memory_properties.memoryTypes[i].heapIndex,
              (unsigned)pdev->memory_properties.memoryTypes[i].propertyFlags,
              (pdev->memory_flags[i] & RADEON_FLAG_32BIT) ? " window32" : "",
              (pdev->memory_flags[i] & RADEON_FLAG_CPU_ACCESS) ? " cpu-visible" : "");
''')],
}

changed = False
for relative, replacements in edits.items():
    path = source / relative
    text = path.read_text()
    if MARK in text:
        continue
    for old, new in replacements:
        if old not in text and new in text:
            continue  # already applied (a file patched before this script marked it)
        if text.count(old) != 1:
            raise SystemExit(f'{relative}: pinned anchor changed: {old.splitlines()[0]!r}')
        text = text.replace(old, new)
    path.write_text(text)
    changed = True
if changed:
    receipt = root.parent / 'mihawk-vulkan-review/.deps/native/radv-release/PROVENANCE.txt'
    if source.resolve() == (root.parent / 'mihawk-vulkan-review/.deps/work/radv-src').resolve() and receipt.exists():
        receipt.replace(receipt.with_suffix('.base.txt'))
print('RADV allocation refusal report: size, window type, pieces and the largest free run, rate-limited')
