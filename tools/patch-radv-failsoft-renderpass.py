#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""A render pass whose attachment allocation failed is skipped, not recorded through NULL.

vk_common_CmdBeginRenderPass2() allocated the per-attachment state array with malloc() and used
it without checking it under NDEBUG; transition_attachment() then read through
cmd_buffer->attachments[att_idx]. A failed allocation now records VK_ERROR_OUT_OF_HOST_MEMORY
and leaves the pass unbegun; transition_attachment() returns 0 when there is no state, and
CmdEndRenderPass2()/CmdNextSubpass2() no-op when no pass was begun. This file's RADV uses the
vk_common entrypoints for render passes, so the guards are live code. Patches the build export,
never the Mesa checkout.
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

BEGIN_OLD = r'''   if (pass->attachment_count > ARRAY_SIZE(cmd_buffer->_attachments)) {
      cmd_buffer->attachments = malloc(pass->attachment_count *
                                       sizeof(*cmd_buffer->attachments));
   } else {
'''
BEGIN_NEW = r'''   if (pass->attachment_count > ARRAY_SIZE(cmd_buffer->_attachments)) {
      cmd_buffer->attachments = malloc(pass->attachment_count *
                                       sizeof(*cmd_buffer->attachments));
      if (cmd_buffer->attachments == NULL) {
         /* Fail soft: a void command cannot report OOM. Record nothing for this pass and leave
          * the command buffer in the error state; End/NextSubpass skip the missing pass. */
         vk_command_buffer_set_error(cmd_buffer, VK_ERROR_OUT_OF_HOST_MEMORY);
         vk_command_buffer_reset_render_pass(cmd_buffer);
         return;
      }
   } else {
'''

TRANSITION_OLD = r'''      &pass->attachments[att_idx];
   struct vk_attachment_state *att_state = &cmd_buffer->attachments[att_idx];
'''
TRANSITION_NEW = r'''      &pass->attachments[att_idx];
   /* A failed attachment allocation left no state to transition (fail-soft BeginRenderPass2). */
   if (cmd_buffer->attachments == NULL)
      return 0;
   struct vk_attachment_state *att_state = &cmd_buffer->attachments[att_idx];
'''

NEXT_OLD = r'''   VK_FROM_HANDLE(vk_command_buffer, cmd_buffer, commandBuffer);

   end_subpass(cmd_buffer, pSubpassEndInfo);
   cmd_buffer->subpass_idx++;
'''
NEXT_NEW = r'''   VK_FROM_HANDLE(vk_command_buffer, cmd_buffer, commandBuffer);

   if (cmd_buffer->render_pass == NULL)
      return;
   end_subpass(cmd_buffer, pSubpassEndInfo);
   cmd_buffer->subpass_idx++;
'''

END_OLD = r'''   VK_FROM_HANDLE(vk_command_buffer, cmd_buffer, commandBuffer);
   const struct vk_render_pass *pass = cmd_buffer->render_pass;
   struct vk_device_dispatch_table *disp =
      &cmd_buffer->base.device->dispatch_table;

   end_subpass(cmd_buffer, pSubpassEndInfo);
'''
END_NEW = r'''   VK_FROM_HANDLE(vk_command_buffer, cmd_buffer, commandBuffer);
   if (cmd_buffer->render_pass == NULL)
      return;
   const struct vk_render_pass *pass = cmd_buffer->render_pass;
   struct vk_device_dispatch_table *disp =
      &cmd_buffer->base.device->dispatch_table;

   end_subpass(cmd_buffer, pSubpassEndInfo);
'''

EDITS = {
    'src/vulkan/runtime/vk_render_pass.c': [
        (BEGIN_OLD, BEGIN_NEW),
        (TRANSITION_OLD, TRANSITION_NEW),
        (NEXT_OLD, NEXT_NEW),
        (END_OLD, END_NEW),
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
            raise SystemExit(f'{relative}: partially applied render-pass fail-soft patch')
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
    print(f'RADV render pass: a failed attachment allocation skips the pass; changed={changed}')
