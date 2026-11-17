#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""What the console's RADV holds of the shared pool, by purpose, and memory asked back on refusal.

The original report of this repository tallied live bytes by RADV's buffer priority (application
memory <= 28, query pools 29, descriptors / upload buffers / fences 30, shaders / scratch 31), and
priority 30 and 31 lump several owners together: on Zelda: Ocarina of Time the p30/p31 split was
the one thing the report could not give (2026-11-17 memory sweep). This report tells the owners
apart by domain, flags and size, as bxymartin/EglantinePS5's (adapted from heydemoura f25ee25):

  app         application memory (VkDeviceMemory, so VMA's blocks)
  query       query pools
  descriptor  descriptor pools (device-local, read-only)
  upload      command buffers' upload buffers (write-combined)
  fence       fences and the other few-byte priority-30 buffers
  gs          a command buffer's 64 MiB heap for geometry shaders run as compute
  shader      shader arenas
  mesh        the emulated task and publish memory (32-bit, 16 MiB or more)
  rings       scratch and the queue's rings

Each purpose prints KiB/buffers/peak/largest/ever made. With RADV_PS5_MEMORY_REPORT=1 (env.txt) a
line goes to stderr at most every 10 s. Without the variable it only keeps the counters.

A refused allocation now asks the app for memory back (eden_memory_recover packs cold guest memory)
up to three rounds, as the app's own allocations do, trying the single run and the pieces after each
round: driver buffers that never pass through VMA (upload buffers, shader arenas, query pools) ended
sessions on the first refusal (memory audit, 2026-11-15). Runs after tools/patch-radv-pieces.py and
tools/patch-radv-alloc-report.py, whose allocation path it changes. Patches the build export, never
the Mesa checkout.
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
MARK = 'EglantinePS5 memory report'

COUNTERS = r'''uint64_t radv_ps5_allocated_bytes;

/* ''' + MARK + r''' (tools/patch-radv-memory-report.py): what the driver holds of the pool by
 * purpose. Priority 30 is descriptor pools, upload buffers, fences and geometry shader heaps;
 * priority 31 shaders, scratch and rings. */
enum {
   RADV_PS5_REPORT_APP,
   RADV_PS5_REPORT_QUERY,
   RADV_PS5_REPORT_DESCRIPTOR,
   RADV_PS5_REPORT_UPLOAD,
   RADV_PS5_REPORT_FENCE,
   RADV_PS5_REPORT_GS,
   RADV_PS5_REPORT_SHADER,
   RADV_PS5_REPORT_MESH,
   RADV_PS5_REPORT_RINGS,
   RADV_PS5_REPORT_PURPOSES,
};
static const char *const radv_ps5_report_names[RADV_PS5_REPORT_PURPOSES] = {
   "app", "query", "descriptor", "upload", "fence", "gs", "shader", "mesh", "rings",
};
static uint64_t radv_ps5_report_bytes[RADV_PS5_REPORT_PURPOSES];
static int32_t radv_ps5_report_buffers[RADV_PS5_REPORT_PURPOSES];
static uint64_t radv_ps5_report_peak[RADV_PS5_REPORT_PURPOSES];
static uint64_t radv_ps5_report_largest[RADV_PS5_REPORT_PURPOSES];
static uint64_t radv_ps5_report_made[RADV_PS5_REPORT_PURPOSES];

static unsigned
radv_ps5_report_purpose(unsigned priority, enum radeon_bo_domain domain, enum radeon_bo_flag flags, uint64_t bytes)
{
   if (priority <= RADV_BO_PRIORITY_APPLICATION_MAX)
      return RADV_PS5_REPORT_APP;
   if (priority == RADV_BO_PRIORITY_QUERY_POOL)
      return RADV_PS5_REPORT_QUERY;
   if (priority == RADV_BO_PRIORITY_DESCRIPTOR) {
      if ((domain & RADEON_DOMAIN_VRAM) && (flags & RADEON_FLAG_READ_ONLY))
         return RADV_PS5_REPORT_DESCRIPTOR;
      if (flags & RADEON_FLAG_GTT_WC)
         return RADV_PS5_REPORT_UPLOAD;
      /* RADV_GS_COMPUTE_HEAP_SIZE, 64 MiB; the others without write combining are a few bytes. */
      return bytes >= (UINT64_C(16) << 20) ? RADV_PS5_REPORT_GS : RADV_PS5_REPORT_FENCE;
   }
   if (!(flags & RADEON_FLAG_32BIT))
      return RADV_PS5_REPORT_RINGS;
   /* A shader arena tops out at 8 MiB. */
   return bytes >= (UINT64_C(16) << 20) ? RADV_PS5_REPORT_MESH : RADV_PS5_REPORT_SHADER;
}

static void
radv_ps5_report_add(unsigned purpose, uint64_t bytes)
{
   const uint64_t held = p_atomic_add_return(&radv_ps5_report_bytes[purpose], bytes);
   p_atomic_inc(&radv_ps5_report_buffers[purpose]);
   p_atomic_inc(&radv_ps5_report_made[purpose]);
   if (held > p_atomic_read(&radv_ps5_report_peak[purpose]))
      p_atomic_set(&radv_ps5_report_peak[purpose], held);
   if (bytes > p_atomic_read(&radv_ps5_report_largest[purpose]))
      p_atomic_set(&radv_ps5_report_largest[purpose], bytes);
}

static void
radv_ps5_report_remove(unsigned purpose, uint64_t bytes)
{
   p_atomic_add(&radv_ps5_report_bytes[purpose], -(int64_t)bytes);
   p_atomic_dec(&radv_ps5_report_buffers[purpose]);
}

void
radv_ps5_memory_report(uint64_t pool, uint64_t available, uint64_t driver)
{
   static int enabled = -1;
   static int64_t last_ns;
   if (enabled < 0) {
      const char *value = getenv("RADV_PS5_MEMORY_REPORT");
      enabled = value && value[0] == '1';
   }
   if (!enabled)
      return;
   const int64_t now = os_time_get_nano();
   const int64_t previous = p_atomic_read(&last_ns);
   if (previous && now - previous < INT64_C(10000000000))
      return;
   if (p_atomic_cmpxchg(&last_ns, previous, now) != previous)
      return;
   char line[1024];
   int length = snprintf(line, sizeof(line), "RADV_PS5_MEMORY pool_mib=%llu available_mib=%llu driver_mib=%llu",
                         (unsigned long long)(pool >> 20), (unsigned long long)(available >> 20),
                         (unsigned long long)(driver >> 20));
   for (unsigned i = 0; i < RADV_PS5_REPORT_PURPOSES && length > 0 && length < (int)sizeof(line) - 96; ++i) {
      if (!p_atomic_read(&radv_ps5_report_made[i]))
         continue;
      length += snprintf(line + length, sizeof(line) - length, " %s=%llu/%d/%llu/%llu/%llu", radv_ps5_report_names[i],
                         (unsigned long long)(p_atomic_read(&radv_ps5_report_bytes[i]) >> 10),
                         (int)p_atomic_read(&radv_ps5_report_buffers[i]),
                         (unsigned long long)(p_atomic_read(&radv_ps5_report_peak[i]) >> 10),
                         (unsigned long long)(p_atomic_read(&radv_ps5_report_largest[i]) >> 10),
                         (unsigned long long)p_atomic_read(&radv_ps5_report_made[i]));
   }
   fprintf(stderr, "%s\n", line);
}
'''

# The report this script replaced (priority-only), for a build export that has it.
OLD_COUNTERS = r'''uint64_t radv_ps5_allocated_bytes;

/* EglantinePS5 memory report: live bytes and buffers by RADV priority. */
static uint64_t radv_ps5_priority_bytes[32];
static int32_t radv_ps5_priority_count[32];

void
radv_ps5_memory_report(uint64_t pool, uint64_t available, uint64_t driver)
{
   static int enabled = -1;
   static int64_t last_ns;
   if (enabled < 0) {
      const char *value = getenv("RADV_PS5_MEMORY_REPORT");
      enabled = value && value[0] == '1';
   }
   if (!enabled)
      return;
   const int64_t now = os_time_get_nano();
   const int64_t previous = p_atomic_read(&last_ns);
   if (previous && now - previous < INT64_C(10000000000))
      return;
   if (p_atomic_cmpxchg(&last_ns, previous, now) != previous)
      return;
   char line[1024];
   int length = snprintf(line, sizeof(line), "RADV_PS5_MEMORY pool_mib=%llu available_mib=%llu driver_mib=%llu",
                         (unsigned long long)(pool >> 20), (unsigned long long)(available >> 20),
                         (unsigned long long)(driver >> 20));
   for (unsigned i = 0; i < 32 && length > 0 && length < (int)sizeof(line) - 48; ++i) {
      const uint64_t bytes = p_atomic_read(&radv_ps5_priority_bytes[i]);
      if (bytes)
         length += snprintf(line + length, sizeof(line) - length, " p%u=%lluMiB/%d", i,
                            (unsigned long long)(bytes >> 20), (int)p_atomic_read(&radv_ps5_priority_count[i]));
   }
   fprintf(stderr, "%s\n", line);
}
'''

ADD = r'''   p_atomic_add(&radv_ps5_allocated_bytes, bo->memory.bytes);
   bo->report_purpose = radv_ps5_report_purpose(priority, domain, flags, bo->memory.bytes);
   radv_ps5_report_add(bo->report_purpose, bo->memory.bytes);
'''
REMOVE = r'''   p_atomic_add(&radv_ps5_allocated_bytes, -(int64_t)bo->memory.bytes);
   radv_ps5_report_remove(bo->report_purpose, bo->memory.bytes);
'''
OLD_ADD = r'''   p_atomic_add(&radv_ps5_allocated_bytes, bo->memory.bytes);
   bo->report_priority = MIN2(priority, 31u);
   p_atomic_add(&radv_ps5_priority_bytes[bo->report_priority], bo->memory.bytes);
   p_atomic_inc(&radv_ps5_priority_count[bo->report_priority]);
'''
OLD_REMOVE = r'''   p_atomic_add(&radv_ps5_allocated_bytes, -(int64_t)bo->memory.bytes);
   p_atomic_add(&radv_ps5_priority_bytes[bo->report_priority], -(int64_t)bo->memory.bytes);
   p_atomic_dec(&radv_ps5_priority_count[bo->report_priority]);
'''

# The allocation fallback as tools/patch-radv-pieces.py leaves it, and as
# tools/patch-radv-alloc-report.py leaves it; both become the retrying form below.
PLAIN_ALLOC = r'''   int64_t physical = -1;
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
'''
REPORTED_ALLOC = r'''   /* ''' + MARK.replace('memory report', 'allocation refusal report') + r''': a refusal says what it asked for and what the pool offered. */
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
'''
RETRY_ALLOC = r'''   /* ''' + MARK.replace('memory report', 'allocation refusal report') + r''': a refusal says what it asked for and what the pool offered. */
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
      /* No run that long: pieces. */
      if (!window32 && report.pieces_wanted &&
          radv_ps5_memory_alloc_pieces(bytes, alignment, out, &report))
         return true;
      /* The app gives memory back when the pool has none (eden_memory_recover packs cold guest
       * memory): up to three rounds, as the app's own allocations do (vulkan_memory_allocator's
       * RecoverDeviceMemory), trying the run and the pieces after each. A driver buffer that never
       * passes through VMA (an upload buffer, a shader arena, a query pool) ended the session on
       * the first refusal before (memory audit, 2026-11-15). */
      while (report.rounds < 3 && eden_memory_recover != NULL) {
         const size_t given = eden_memory_recover(bytes + (UINT64_C(64) << 20));
         if (given == 0)
            break;
         report.recovered += given;
         report.rounds++;
         run_result = sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), bytes, alignment,
                                                   RADV_PS5_DIRECT_TYPE, &physical);
         report.run_result = run_result;
         if (run_result == 0)
            break;
         if (!window32 && report.pieces_wanted &&
             radv_ps5_memory_alloc_pieces(bytes, alignment, out, &report))
            return true;
      }
      if (run_result != 0) {
         radv_ps5_alloc_refused(&report);
         return false;
      }
   }
'''

changed = False

# radv_ps5_winsys.h: the buffer's purpose, from the previous report's priority field if present.
path = source / 'src/amd/vulkan/winsys/ps5/radv_ps5_winsys.h'
text = path.read_text()
if 'uint8_t report_purpose;' not in text:
    old_field = '   /* RADV\'s priority for the buffer (EglantinePS5 memory report). */\n   uint8_t report_priority;\n'
    new_field = '   /* What the buffer is for (EglantinePS5 memory report). */\n   uint8_t report_purpose;\n'
    if old_field in text:
        text = text.replace(old_field, new_field)
    elif '   bool imported;\n};\n' in text:
        text = text.replace('   bool imported;\n};\n',
                            '   bool imported;\n' + new_field + '};\n\n'
                            'void radv_ps5_memory_report(uint64_t pool, uint64_t available, uint64_t driver);\n')
    else:
        raise SystemExit('radv_ps5_winsys.h: pinned anchor changed: bool imported')
    path.write_text(text)
    changed = True

# radv_ps5_bo.c: the counters, the report and its two updates.
path = source / 'src/amd/vulkan/winsys/ps5/radv_ps5_bo.c'
text = path.read_text()
if 'RADV_PS5_REPORT_PURPOSES' not in text:
    if OLD_COUNTERS in text:
        text = text.replace(OLD_COUNTERS, COUNTERS)
        text = text.replace(OLD_ADD, ADD)
        text = text.replace(OLD_REMOVE, REMOVE)
        if OLD_ADD in text or OLD_REMOVE in text:
            raise SystemExit('radv_ps5_bo.c: the previous report did not match')
    else:
        text = text.replace('uint64_t radv_ps5_allocated_bytes;\n', COUNTERS)
        text = text.replace('   p_atomic_add(&radv_ps5_allocated_bytes, bo->memory.bytes);\n', ADD)
        text = text.replace('   p_atomic_add(&radv_ps5_allocated_bytes, -(int64_t)bo->memory.bytes);\n', REMOVE)
        if COUNTERS not in text or ADD not in text or REMOVE not in text:
            raise SystemExit('radv_ps5_bo.c: pinned anchor changed: allocated_bytes')
    path.write_text(text)
    changed = True

# radv_physical_device.c: call the report at every memory budget query. tools/patch-radv-wsi.py
# rewrites this file from the pinned source each run, so the call is added again here.
path = source / 'src/amd/vulkan/radv_physical_device.c'
text = path.read_text()
if 'radv_ps5_memory_report(pool, available, driver);' not in text:
    old = '   radv_ps5_winsys_heap_usage(&pool, &available, &driver);\n'
    if old not in text:
        raise SystemExit('radv_physical_device.c: pinned anchor changed: heap usage')
    text = text.replace(old, old + '   void radv_ps5_memory_report(uint64_t, uint64_t, uint64_t); /* ' + MARK + ' */\n'
                             '   radv_ps5_memory_report(pool, available, driver);\n')
    path.write_text(text)
    changed = True

# radv_ps5_platform.c: three rounds of memory asked back on a refusal.
path = source / 'src/amd/vulkan/winsys/ps5/radv_ps5_platform.c'
text = path.read_text()
if 'while (report.rounds < 3 && eden_memory_recover != NULL)' not in text:
    if REPORTED_ALLOC in text:
        text = text.replace(REPORTED_ALLOC, RETRY_ALLOC)
    elif PLAIN_ALLOC in text:
        text = text.replace(PLAIN_ALLOC, RETRY_ALLOC)
    else:
        raise SystemExit('radv_ps5_platform.c: pinned anchor changed: the allocation fallback')
    path.write_text(text)
    changed = True

if changed:
    receipt = root.parent / 'mihawk-vulkan-review/.deps/native/radv-release/PROVENANCE.txt'
    if source.resolve() == (root.parent / 'mihawk-vulkan-review/.deps/work/radv-src').resolve() and receipt.exists():
        receipt.replace(receipt.with_suffix('.base.txt'))
print('RADV memory report: by purpose (RADV_PS5_MEMORY_REPORT=1), three rounds of memory asked back')
