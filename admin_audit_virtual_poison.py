#!/usr/bin/env python3
"""
admin_audit_virtual_poison.py — monthly read-only audit for mislabeled crops
============================================================================
Counts colourful `__empty__` / `__inactive__` crops in `data/` that are NOT
yet resolved in the review ledger (`data/reviewed_virtual.jsonl`). These are
exactly the crops the client logs as `CommunitySeed: POISON skip` and that
`admin_reject_crops.py` is built to review/reject/relabel.

Healthy steady state: 0 — every colourful virtual crop has been reviewed
(KEEP) or removed/relabeled. Anything above `--max` exits 1 so the scheduled
GitHub workflow fails and emails the repo owner (same pattern as
`admin_audit_staging.py`). We do NOT auto-fix: cleanup is a manual review via
`admin_reject_crops.py` (or the maintainer console) after eyeballing the crops.

Read-only. Never commits. Reuses `admin_reject_crops.scan` so the audit's
notion of "poison" is byte-for-byte the same as the review tool and the
client-side guard.

Usage:
    .venv/bin/python admin_audit_virtual_poison.py            # BREACH if > 0
    .venv/bin/python admin_audit_virtual_poison.py --max 5

Environment (.env, same as the mergers):
    HF_TOKEN     — read token (any HF token works for cloning)
    HF_DATASET   — default: sets-sto/sto-icon-dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from admin_reject_crops import (  # noqa: E402
    REPO, RTYPE, HF_TOKEN, VIRTUAL_SEED_BRIGHT_RATIO, VIRTUAL_SEED_RICH_RATIO,
    scan,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--max', type=int, default=0,
                    help='Max unreviewed colourful virtual crops before BREACH '
                         '(default: 0 — every one must be reviewed).')
    ap.add_argument('--bright-ratio', type=float, default=VIRTUAL_SEED_BRIGHT_RATIO,
                    help='Bright-pixel gate (keep in sync with the client).')
    ap.add_argument('--rich-ratio', type=float, default=VIRTUAL_SEED_RICH_RATIO,
                    help='Colour-rich gate (keep in sync with the client).')
    args = ap.parse_args()

    if not HF_TOKEN:
        print('ERROR: HF_TOKEN is empty (set in .env or shell).', file=sys.stderr)
        return 2

    print(f'== Dataset repo: {REPO}')
    from hf_clone import clone_hf_shallow
    snap_dir = clone_hf_shallow(REPO, HF_TOKEN, repo_type=RTYPE)

    candidates = scan(snap_dir, HF_TOKEN, args.bright_ratio, args.rich_ratio,
                      show_reviewed=False)
    n = len(candidates)
    status = 'BREACH' if n > args.max else 'OK'

    print('\n─── Audit result ─────────────────────────────────────')
    print(f'AUDIT: domain=virtual_poison unreviewed={n:<6} '
          f'threshold={args.max:<6} status={status}')
    for e in candidates[:20]:
        print(f'  {e["sha"][:10]} {e["name"]:<12} slot={e["slot"]!r:<18} '
              f'bright={e["bright"]:.1%} rich={e["rich"]:.1%}')
    if n > 20:
        print(f'  … and {n - 20} more')

    if n > args.max:
        print(f'\nFAIL: {n} unreviewed colourful virtual crop(s) exceed '
              f'max={args.max}.')
        print('      Review with `admin_reject_crops.py` (or the maintainer '
              'console) — reject, relabel, or KEEP each one.')
        return 1
    print('\nOK: no unreviewed colourful virtual crops above threshold.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
