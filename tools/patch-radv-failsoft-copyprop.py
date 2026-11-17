#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Copy propagation drops a copy it cannot allocate instead of writing through NULL.

copy_entry_create() appended to a util_dynarray with util_dynarray_append(), which memcpy()s
through the array's data pointer; under NDEBUG a failed reallocation left that pointer NULL (or
the grow unchecked) and the pass wrote through it. It now uses util_dynarray_grow(), returns NULL
when the grow fails, and all three call sites skip that instruction's copy; the pass only loses
an optimization. Patches the build export, never the Mesa checkout.
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

CREATE_OLD = r'''   struct copy_entry new_entry = {
      .dst = *deref,
   };
   util_dynarray_append(copies_array, new_entry);
   return util_dynarray_top_ptr(copies_array, struct copy_entry);
'''
CREATE_NEW = r'''   struct copy_entry new_entry = {
      .dst = *deref,
   };
   struct copy_entry *entry =
      util_dynarray_grow(copies_array, struct copy_entry, 1);
   if (entry == NULL) {
      /* Fail soft: out of memory. Drop this copy instead of writing through a null pointer; the
       * pass only loses an optimization. */
      return NULL;
   }
   *entry = new_entry;
   return entry;
'''

LOAD_OLD = r'''         if (!entry)
            entry = copy_entry_create(state, copies, &vec_src);

         /* Update the entry with the value of the load.  This way
'''
LOAD_NEW = r'''         if (!entry)
            entry = copy_entry_create(state, copies, &vec_src);
         if (entry == NULL)
            break;

         /* Update the entry with the value of the load.  This way
'''

STORE_OLD = r'''            struct copy_entry *entry =
               get_entry_and_kill_aliases(state, copies, &vec_dst, wrmask);
            value_set_from_value(&entry->src, &value, vec_index, wrmask);
'''
STORE_NEW = r'''            struct copy_entry *entry =
               get_entry_and_kill_aliases(state, copies, &vec_dst, wrmask);
            if (entry == NULL)
               break;
            value_set_from_value(&entry->src, &value, vec_index, wrmask);
'''

COPY_OLD = r'''         struct copy_entry *dst_entry =
            get_entry_and_kill_aliases(state, copies, &dst, full_mask);
         value_set_from_value(&dst_entry->src, &value, 0, full_mask);
'''
COPY_NEW = r'''         struct copy_entry *dst_entry =
            get_entry_and_kill_aliases(state, copies, &dst, full_mask);
         if (dst_entry == NULL)
            break;
         value_set_from_value(&dst_entry->src, &value, 0, full_mask);
'''

EDITS = {
    'src/compiler/nir/nir_opt_copy_prop_vars.c': [
        (CREATE_OLD, CREATE_NEW),
        (LOAD_OLD, LOAD_NEW),
        (STORE_OLD, STORE_NEW),
        (COPY_OLD, COPY_NEW),
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
            raise SystemExit(f'{relative}: partially applied copy-propagation fail-soft patch')
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
    print(f'NIR copy propagation: a failed copy allocation drops the copy; changed={changed}')
