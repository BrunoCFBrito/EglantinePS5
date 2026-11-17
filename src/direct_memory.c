// SPDX-License-Identifier: GPL-3.0-or-later
// Where the app's long-lived memory goes in the pool the CPU shares with the GPU.
//
// The system gives out direct memory from the bottom of the pool: the lowest room that fits.
// A game's own memory (4 GiB) and its graphics come and go with each session, while two kinds
// of blocks stay for good: the pieces the heap grows by (headless/heap_arenas.inc) and the
// stacks of threads (the Payload SDK takes 2 MiB of direct memory for each, and keeps freed
// ones for the next threads). Taken from the bottom like the rest, those landed between a
// running game's graphics memory, and when the game ended they were left spread over the
// middle of the pool. The next game's graphics get the largest free stretch, which was then
// much smaller than what was free in total, and smaller again after every game.
//
// So blocks that stay are taken from the top of the pool downwards, and everything a session
// takes and gives back is left to the system as before. The link wraps the calls
// (tools/link-headless-native.sh): the heap asks for the top itself (eden_direct_high), and
// the other blocks that stay are known by their size: a thread's stack is 2 MiB, and so is the
// one block the graphics driver keeps for good (its zero block, taken in the middle of a game
// and then left where it was). The driver's command ring and marker page are known by who
// asks. The app's own 2 MiB blocks, the slots of the guest page tables, go back when a game
// ends: they ask below the wrap (eden_direct_low) and stay where the system puts them.
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

int __real_sceKernelAllocateDirectMemory(int64_t, int64_t, size_t, size_t, int, int64_t *);
int __real_sceKernelReleaseDirectMemory(int64_t, size_t);
int64_t sceKernelGetDirectMemorySize(void);

#define EDEN_DIRECT_STEP ((int64_t)2 << 20)       /* blocks at the top are placed in 2 MiB steps */
#define EDEN_DIRECT_SMALL ((size_t)2 << 20)       /* a thread's stack; the driver's zero block */

static atomic_flag eden_direct_lock = ATOMIC_FLAG_INIT;
static int64_t eden_direct_floor;                 /* the lowest block taken at the top; 0: none yet */
static int64_t eden_direct_holes[1024];           /* 2 MiB blocks given back above the floor */
static unsigned eden_direct_hole_count;

static void eden_direct_acquire(void) {
  while (atomic_flag_test_and_set_explicit(&eden_direct_lock, memory_order_acquire)) {}
}
static void eden_direct_release(void) { atomic_flag_clear_explicit(&eden_direct_lock, memory_order_release); }

/* Who takes and gives back direct memory, by call site, printed with EDEN_MEMORY every 10 s
 * (eden_direct_report): a pool split into hundreds of holes is made by many small gives back
 * between blocks that stay. The cost of placing blocks at the top is counted too: every try
 * there is a call to the system. */
#define EDEN_DIRECT_SITES 64
struct eden_direct_site {
  uintptr_t caller;
  unsigned long count;
  unsigned long bytes;
};
static struct eden_direct_site eden_direct_takes[EDEN_DIRECT_SITES], eden_direct_gives[EDEN_DIRECT_SITES];
static atomic_flag eden_direct_sites_lock = ATOMIC_FLAG_INIT;
static atomic_ulong eden_direct_high_calls, eden_direct_high_tries, eden_direct_high_ns, eden_direct_take_rounds;

static void eden_direct_note(struct eden_direct_site *sites, const void *caller, size_t size) {
  const uintptr_t at = (uintptr_t)caller;
  while (atomic_flag_test_and_set_explicit(&eden_direct_sites_lock, memory_order_acquire)) {}
  for (unsigned i = 0; i < EDEN_DIRECT_SITES; ++i) {
    if (sites[i].caller == at || sites[i].caller == 0 || i == EDEN_DIRECT_SITES - 1) {
      sites[i].caller = sites[i].caller == 0 ? at : sites[i].caller; /* the last entry takes the rest */
      sites[i].count++;
      sites[i].bytes += size;
      break;
    }
  }
  atomic_flag_clear_explicit(&eden_direct_sites_lock, memory_order_release);
}

static unsigned long eden_direct_now_ns(void) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (unsigned long)now.tv_sec * 1000000000ul + (unsigned long)now.tv_nsec;
}

static void eden_direct_print(FILE *out, const char *name, struct eden_direct_site *sites) {
  struct eden_direct_site copy[EDEN_DIRECT_SITES];
  while (atomic_flag_test_and_set_explicit(&eden_direct_sites_lock, memory_order_acquire)) {}
  for (unsigned i = 0; i < EDEN_DIRECT_SITES; ++i) copy[i] = sites[i];
  atomic_flag_clear_explicit(&eden_direct_sites_lock, memory_order_release);
  for (unsigned shown = 0; shown < 8; ++shown) {
    unsigned best = EDEN_DIRECT_SITES;
    for (unsigned i = 0; i < EDEN_DIRECT_SITES; ++i)
      if (copy[i].count != 0 && (best == EDEN_DIRECT_SITES || copy[i].count > copy[best].count)) best = i;
    if (best == EDEN_DIRECT_SITES) break;
    /* Offsets as EDEN_HEAP_SITE prints them (the code is loaded at 0x400000). */
    fprintf(out, "%s n=%lu MiB=%lu site=%lx\n", name, copy[best].count, copy[best].bytes >> 20,
            (unsigned long)(copy[best].caller - 0x400000ul));
    copy[best].count = 0;
  }
}

void eden_direct_report(FILE *out) {
  fprintf(out, "EDEN_DIRECT top_placements=%lu tries=%lu ms=%lu retry_rounds=%lu\n",
          atomic_load_explicit(&eden_direct_high_calls, memory_order_relaxed),
          atomic_load_explicit(&eden_direct_high_tries, memory_order_relaxed),
          atomic_load_explicit(&eden_direct_high_ns, memory_order_relaxed) / 1000000ul,
          atomic_load_explicit(&eden_direct_take_rounds, memory_order_relaxed));
  eden_direct_print(out, "EDEN_DIRECT_TAKE", eden_direct_takes);
  eden_direct_print(out, "EDEN_DIRECT_GIVE", eden_direct_gives);
}

/* Memory from as high in the pool as there is room; from anywhere when the top has none. */
static int eden_direct_high_place(size_t size, size_t alignment, int type, int64_t *start) {
  const int64_t total = sceKernelGetDirectMemorySize();
  const int64_t bytes = (int64_t)size;
  if (bytes <= 0 || total <= 0) return __real_sceKernelAllocateDirectMemory(0, total, size, alignment, type, start);
  eden_direct_acquire();
  /* A block given back earlier left a hole of its size. */
  while (size == EDEN_DIRECT_SMALL && eden_direct_hole_count != 0) {
    const int64_t hole = eden_direct_holes[--eden_direct_hole_count];
    if (__real_sceKernelAllocateDirectMemory(hole, hole + bytes, size, alignment, type, start) == 0) {
      eden_direct_release();
      return 0;
    }
  }
  /* Below what is at the top already; past what something else has put there, for a while. */
  int64_t next = eden_direct_floor != 0 ? eden_direct_floor : total;
  for (unsigned tries = 0; tries < 512 && next >= bytes; ++tries) {
    const int64_t at = (next - bytes) / EDEN_DIRECT_STEP * EDEN_DIRECT_STEP;
    next = at;
    atomic_fetch_add_explicit(&eden_direct_high_tries, 1, memory_order_relaxed);
    if (__real_sceKernelAllocateDirectMemory(at, at + bytes, size, alignment, type, start) == 0) {
      eden_direct_floor = at;
      eden_direct_release();
      return 0;
    }
  }
  eden_direct_release();
  return __real_sceKernelAllocateDirectMemory(0, total, size, alignment, type, start);
}

static int eden_direct_high_timed(size_t size, size_t alignment, int type, int64_t *start) {
  const unsigned long began = eden_direct_now_ns();
  const int result = eden_direct_high_place(size, alignment, type, start);
  atomic_fetch_add_explicit(&eden_direct_high_calls, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&eden_direct_high_ns, eden_direct_now_ns() - began, memory_order_relaxed);
  return result;
}

int eden_direct_high(size_t size, size_t alignment, int type, int64_t *start) {
  eden_direct_note(eden_direct_takes, __builtin_return_address(0), size);
  return eden_direct_high_timed(size, alignment, type, start);
}

/* As the system places it: for what a session takes and gives back. */
int eden_direct_low(size_t size, size_t alignment, int type, int64_t *start) {
  eden_direct_note(eden_direct_takes, __builtin_return_address(0), size);
  return __real_sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), size, alignment, type, start);
}

/* The Vulkan driver makes its command ring (16 MiB) and, right after it, the ring's marker page
 * once, with the first device, and keeps them until the app ends. Taken from the bottom like a
 * session's memory they landed above the first game's (the ring at 4.6 GiB), and every later
 * game found the pool's largest free stretch cut from 10.3 GiB to 5.7. They are known by who
 * asks (the driver's allocator; no such symbol in a build without that driver), by their sizes
 * and by being the first of those sizes in that order. */
extern void eden_radv_private_radv_ps5_memory_alloc(void) __attribute__((weak));
#define EDEN_DIRECT_RING ((size_t)16 << 20)
#define EDEN_DIRECT_PAGE ((size_t)16 << 10)
static atomic_int eden_direct_driver_kept; /* 0: neither yet; 1: the ring; 2: the marker page too */

static int eden_direct_driver_keeps(size_t size, const void *caller) {
  const char *const allocator = (const char *)(uintptr_t)&eden_radv_private_radv_ps5_memory_alloc;
  if (!allocator || (const char *)caller < allocator || (const char *)caller >= allocator + 2048) return 0;
  int kept = atomic_load_explicit(&eden_direct_driver_kept, memory_order_relaxed);
  if (kept == 0 && size == EDEN_DIRECT_RING)
    return atomic_compare_exchange_strong(&eden_direct_driver_kept, &kept, 1);
  if (kept == 1 && size == EDEN_DIRECT_PAGE)
    return atomic_compare_exchange_strong(&eden_direct_driver_kept, &kept, 2);
  return 0;
}

/* src/guest_memory_service.inc: cold guest memory out now, if the swap is free (bytes given back);
 * wake the swapper. Weak: absent in builds without the service. */
extern size_t eden_memory_recover_try(size_t want) __attribute__((weak));
extern void eden_guest_memory_urgent(void) __attribute__((weak));

/* A block that stays (a thread's stack, the driver's zero block, ring and marker page) the pool
 * refused: memory is made and it is asked again, up to 8 times. A refused stack ended the app
 * (std::thread throws; memory audit, 2026-11-15). Only these: the driver and the heap first ask for
 * one large range and fall back to pieces when it is refused, and a retry here would hold every such
 * request up. One thread at a time makes memory here; the others ask again after it. The swap's own
 * blocks never come this way (eden_direct_low), so it cannot recurse. */
static atomic_int eden_direct_making;
static int eden_direct_high_take(size_t size, size_t alignment, int type, int64_t *start) {
  int result = eden_direct_high_timed(size, alignment, type, start);
  for (unsigned round = 0; result != 0 && round < 8 && eden_memory_recover_try != NULL; ++round) {
    atomic_fetch_add_explicit(&eden_direct_take_rounds, 1, memory_order_relaxed);
    if (eden_guest_memory_urgent != NULL) eden_guest_memory_urgent();
    if (atomic_fetch_add_explicit(&eden_direct_making, 1, memory_order_acq_rel) == 0)
      (void)eden_memory_recover_try(size + ((size_t)16 << 20));
    atomic_fetch_sub_explicit(&eden_direct_making, 1, memory_order_acq_rel);
    const struct timespec pause = {0, 10000000};
    nanosleep(&pause, NULL);
    result = eden_direct_high_timed(size, alignment, type, start);
  }
  return result;
}

int __wrap_sceKernelAllocateDirectMemory(int64_t from, int64_t to, size_t size, size_t alignment, int type,
                                         int64_t *start) {
  eden_direct_note(eden_direct_takes, __builtin_return_address(0), size);
  /* A thread's stack or the driver's zero block, asked for anywhere in the pool; and the
   * driver's command ring and marker page. */
  if (from == 0 && start && to >= sceKernelGetDirectMemorySize() &&
      (size == EDEN_DIRECT_SMALL || eden_direct_driver_keeps(size, __builtin_return_address(0))))
    return eden_direct_high_take(size, alignment, type, start);
  return __real_sceKernelAllocateDirectMemory(from, to, size, alignment, type, start);
}

int __wrap_sceKernelReleaseDirectMemory(int64_t start, size_t size) {
  const int result = __real_sceKernelReleaseDirectMemory(start, size);
  if (result != 0) return result;
  eden_direct_note(eden_direct_gives, __builtin_return_address(0), size);
  eden_direct_acquire();
  if (size == EDEN_DIRECT_SMALL && eden_direct_floor != 0 && start >= eden_direct_floor &&
      eden_direct_hole_count < sizeof(eden_direct_holes) / sizeof(eden_direct_holes[0]))
    eden_direct_holes[eden_direct_hole_count++] = start;
  eden_direct_release();
  return result;
}
