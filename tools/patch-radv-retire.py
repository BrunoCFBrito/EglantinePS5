#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""A freed buffer object keeps its contents and address for a slack before its pages go back.

Command lists recorded before a free may still run (Eden's split GPU thread is frames behind),
so radv_ps5_memory_free() hands an eligible freed allocation (one whole direct-memory allocation,
16 KiB aligned) to a retire list instead of unmapping it at once. radv_ps5_eden_trim(want)
releases retired allocations oldest-first, RADV_PS5_PAGE_BYTES (16 KiB) at a time, up to `want`;
each page is mapped to the shared zero block before its direct memory is released, so a late
access reads zeros instead of a reallocated page or a fault. RADV_PS5_RETIRE_SLACK (ms, default
2000) and RADV_PS5_RETIRE_MAX_MIB (MiB, default 256; 0 frees at once, as before) tune it.
Piecewise and replay allocations are freed at once: they are several allocations or not ours to
hold. Applied after patch-radv-trim.py, whose radv_ps5_trim_untrack and radv_ps5_eden_trim it
extends. Patches the build export, never the Mesa checkout.
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

MARK = 'EglantinePS5 retire'

RETIRE = r'''
/* ''' + MARK + r''' (agent-radv-retire.patch): a freed buffer object keeps its contents and
 * address for RADV_PS5_RETIRE_SLACK ms. Command lists recorded before the free may still run (the
 * split GPU thread is frames behind); radv_ps5_eden_trim then gives the allocation's pages back
 * 16 KiB at a time, oldest first, when the app asks for memory. Each page is mapped to the shared
 * zero block before its direct memory is released, so a late access reads zeros instead of a
 * reallocated page or a fault. RADV_PS5_RETIRE_SLACK (ms, default 2000) and
 * RADV_PS5_RETIRE_MAX_MIB (MiB, default 256; 0 frees at once as before) tune it. Piecewise and
 * replay allocations are freed at once: they are several allocations or not ours to hold. */
struct radv_ps5_retired {
   struct radv_ps5_memory memory;
   struct radv_ps5_retired *next;
   uint32_t retired_ms;
   uint64_t given_back;
};
static struct radv_ps5_retired *radv_ps5_retire_head, *radv_ps5_retire_tail;
static uint32_t radv_ps5_retire_count;
static uint64_t radv_ps5_retire_bytes, radv_ps5_retire_returned;

static uint32_t
radv_ps5_retire_slack_ms(void)
{
   static uint32_t value;
   if (!value) {
      const char *text = getenv("RADV_PS5_RETIRE_SLACK");
      value = text ? (uint32_t)strtoul(text, NULL, 10) : 2000u;
      if (!value)
         value = 1;
   }
   return value;
}

static uint64_t
radv_ps5_retire_max_bytes(void)
{
   static bool parsed;
   static uint64_t value;
   if (!parsed) {
      const char *text = getenv("RADV_PS5_RETIRE_MAX_MIB");
      value = text ? (uint64_t)strtoull(text, NULL, 10) << 20 : UINT64_C(256) << 20;
      parsed = true;
   }
   return value;
}

/* Takes a freed allocation; false: free it at once. The trim lock orders it with trim. */
static bool
radv_ps5_memory_retire(struct radv_ps5_memory *memory)
{
   if (memory->window_replay || memory->pieces || memory->physical < 0 || memory->bytes == 0 ||
       memory->bytes % RADV_PS5_PAGE_BYTES || (uintptr_t)memory->cpu % RADV_PS5_PAGE_BYTES)
      return false;
   if (radv_ps5_retire_max_bytes() == 0)
      return false;
   struct radv_ps5_retired *const node = malloc(sizeof(*node));
   if (!node)
      return false;
   radv_ps5_trim_acquire();
   if (radv_ps5_retire_bytes + memory->bytes > radv_ps5_retire_max_bytes()) {
      radv_ps5_trim_release();
      free(node);
      return false;
   }
   node->memory = *memory;
   node->retired_ms = radv_ps5_trim_now_ms();
   node->given_back = 0;
   node->next = NULL;
   if (radv_ps5_retire_tail)
      radv_ps5_retire_tail->next = node;
   else
      radv_ps5_retire_head = node;
   radv_ps5_retire_tail = node;
   radv_ps5_retire_count++;
   radv_ps5_retire_bytes += memory->bytes;
   radv_ps5_trim_release();
   *memory = (struct radv_ps5_memory){.physical = -1};
   return true;
}

/* Gives back up to `want` bytes (0: all) of the oldest retired allocation, 16 KiB at a time. The
 * trim lock is held. Returns what it took back; the node is freed when nothing is left. */
static uint64_t
radv_ps5_retire_give_back(uint64_t want)
{
   struct radv_ps5_retired *const node = radv_ps5_retire_head;
   const int64_t zero = radv_ps5_zero_block();
   if (zero < 0)
      return 0;
   struct radv_ps5_memory *const memory = &node->memory;
   uint64_t returned = 0;
   while (node->given_back < memory->bytes && (!want || returned < want)) {
      uint8_t *const at = memory->cpu + node->given_back;
      const int64_t physical = memory->physical + (int64_t)node->given_back;
      if (!radv_ps5_map_fixed(at, RADV_PS5_PAGE_BYTES, zero))
         break;
      if (sceKernelReleaseDirectMemory(physical, RADV_PS5_PAGE_BYTES) != 0) {
         /* Kept: its page goes back in place, and the rest waits for the next ask. */
         if (!radv_ps5_map_fixed(at, RADV_PS5_PAGE_BYTES, physical))
            abort();
         break;
      }
      node->given_back += RADV_PS5_PAGE_BYTES;
      returned += RADV_PS5_PAGE_BYTES;
   }
   radv_ps5_retire_returned += returned;
   radv_ps5_retire_bytes -= returned;
   if (node->given_back == memory->bytes) {
      const int32_t unmapped = sceKernelMunmap(memory->cpu, memory->bytes);
      if (unmapped != 0)
         fprintf(stderr, "radv/ps5: retired range at %p could not be unmapped: 0x%08x\n",
                 (void *)memory->cpu, (unsigned)unmapped);
      radv_ps5_granules_give(&radv_ps5_region, memory->granule, memory->granules);
      radv_ps5_retire_head = node->next;
      if (!radv_ps5_retire_head)
         radv_ps5_retire_tail = NULL;
      radv_ps5_retire_count--;
      free(node);
   }
   return returned;
}

'''

TRIM_LOOP = r'''   /* Freed buffer objects whose slack has passed go back 16 KiB at a time, oldest first. */
   const uint32_t retire_slack = radv_ps5_retire_slack_ms();
   const uint64_t retire_before = radv_ps5_retire_returned;
   while (zero >= 0 && radv_ps5_retire_head &&
          now - radv_ps5_retire_head->retired_ms >= retire_slack && (!want || returned < want)) {
      const uint64_t taken = radv_ps5_retire_give_back(want ? want - returned : 0);
      if (taken == 0)
         break;
      returned += taken;
   }
   static uint32_t retire_told;
   const uint64_t retired = radv_ps5_retire_returned - retire_before;
   if (retired && retire_told++ < 16)
      fprintf(stderr, "radv/ps5: retire gave back %" PRIu64 " MiB in 16 KiB pages (%" PRIu64
              " MiB retired, %u buffer objects waiting)\n", retired >> 20,
              radv_ps5_retire_bytes >> 20, radv_ps5_retire_count);

'''

EDITS = {
    'src/amd/vulkan/winsys/ps5/radv_ps5_platform.c': [
        ('static void\nradv_ps5_trim_untrack(struct radv_ps5_memory *memory)\n{',
         RETIRE + 'static void\nradv_ps5_trim_untrack(struct radv_ps5_memory *memory)\n{'),
        ('   }\n   radv_ps5_trim_returned += returned;\n',
         '   }\n' + TRIM_LOOP + '   radv_ps5_trim_returned += returned;\n'),
        ('   radv_ps5_trim_untrack(memory);\n   const int32_t unmapped = sceKernelMunmap(memory->cpu, memory->bytes);\n',
         '   radv_ps5_trim_untrack(memory);\n   if (radv_ps5_memory_retire(memory))\n      return;\n'
         '   const int32_t unmapped = sceKernelMunmap(memory->cpu, memory->bytes);\n'),
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
            raise SystemExit(f'{relative}: partially applied retire patch')
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
        # receipt aside so the next build compiles the changed file (Ninja keeps every other
        # object).
        receipt = root.parent / 'mihawk-vulkan-review/.deps/native/radv-release/PROVENANCE.txt'
        if source.resolve() == (root.parent / 'mihawk-vulkan-review/.deps/work/radv-src').resolve() and receipt.exists():
            receipt.replace(receipt.with_suffix('.base.txt'))
    print(f'RADV retire: freed buffer objects keep their pages for a slack; changed={changed}')
