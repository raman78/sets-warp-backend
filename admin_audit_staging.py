#!/usr/bin/env python3
"""
admin_audit_staging.py — Read-only health check for staging vs data/
======================================================================

Sibling of admin_drain_stale_staging.py — same orphan-detection logic,
but **counts only, never deletes**. Designed to run on a monthly cron
(see .github/workflows/audit_staging_health.yml). If any domain's
orphan count exceeds its threshold, the script exits 1 — GitHub Actions
turns that into a failed scheduled workflow and emails the repo owner.

The philosophy: regular mergers drain on promote; this audit catches
the leak when they don't. We do NOT auto-fix — surfacing the anomaly
forces a root-cause look instead of papering over it.

Definitions:
    "orphan" — a staging file whose semantic key is already in data/
    (so the new merger will never look at it again — no fresh votes).

    Per domain:
      crops          staging/<iid>/crops/<sha>.png            sha ∈ data/annotations.jsonl
      screens        staging/<iid>/screen_types/<T>/<sha>.png sha ∈ data/screen_types/metadata.jsonl
      contributions  contributions/<date>/<id>.json           id  ∈ knowledge.json::processed_contributions
      anchors (opt)  staging/<iid>/anchors_grid_*.json        (build_type, aspect_bucket) ∈ data/anchors/*.json

Output:
    Human-readable summary table to stdout.
    Machine-readable `AUDIT:` line per domain, grep-friendly.
    Exit 0  — all domains under their threshold.
    Exit 1  — at least one threshold breached.

Usage:
    .venv/bin/python admin_audit_staging.py                    # default thresholds
    .venv/bin/python admin_audit_staging.py --crops-max 50     # tighter
    .venv/bin/python admin_audit_staging.py --include-anchors  # audit anchors too

Environment (.env, same as the mergers):
    HF_TOKEN     — read token (any HF token works for listing)
    HF_REPO_ID   — default: sets-sto/warp-knowledge
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Reuse the consensus-loading + plan-building helpers — single source of
# truth so the audit and the drain agree on what counts as an orphan.
from admin_drain_stale_staging import (
    HF_DATASET, HF_KNOW, HF_TOKEN,
    _list_repo_files,
    _load_jsonl_field,
    _load_knowledge_processed,
    _load_anchor_keys,
    _plan_crops_drain,
    _plan_screens_drain,
    _plan_anchors_drain,
    _plan_contributions_drain,
)


def _emit_audit_line(domain: str, orphans: int, threshold: int) -> bool:
    """One grep-friendly line per domain. Returns True iff threshold breached."""
    status = 'BREACH' if orphans > threshold else 'OK'
    print(f'AUDIT: domain={domain:<13} orphans={orphans:<6} '
          f'threshold={threshold:<6} status={status}')
    return orphans > threshold


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--crops-max',    type=int, default=100,
                    help='Max crop-PNG orphans before BREACH (default: 100)')
    ap.add_argument('--screens-max',  type=int, default=50,
                    help='Max screen-PNG orphans before BREACH (default: 50)')
    ap.add_argument('--contribs-max', type=int, default=50,
                    help='Max processed-but-undrained contributions (default: 50)')
    ap.add_argument('--anchors-max',  type=int, default=20,
                    help='Max stale anchor files (default: 20, only used '
                         'with --include-anchors)')
    ap.add_argument('--include-anchors', action='store_true',
                    help='Audit staging anchor files too. Off by default '
                         'because anchors aggregate votes per file — a '
                         '"stale" anchor with new contributors should not '
                         'trip the alert until the merger has had a cycle '
                         'to absorb them.')
    args = ap.parse_args()

    if not HF_TOKEN:
        print('ERROR: HF_TOKEN is empty (set in .env or shell).', file=sys.stderr)
        return 2

    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)

    print(f'== Dataset repo: {HF_DATASET}')
    repo_files = _list_repo_files(api, HF_DATASET, 'dataset')
    print(f'   {len(repo_files)} paths listed.')

    print('-- Loading consensus sets …')
    promoted_crops   = _load_jsonl_field(api, HF_DATASET,
                                         'data/annotations.jsonl',
                                         'crop_sha256')
    promoted_screens = _load_jsonl_field(api, HF_DATASET,
                                         'data/screen_types/metadata.jsonl',
                                         'sha')
    promoted_anchors: set[tuple[str, str]] = set()
    if args.include_anchors:
        promoted_anchors = _load_anchor_keys(api, repo_files)

    print('-- Counting orphans …')
    # Reuse the drain planner — same logic, we just discard the plan and
    # count what would have been deleted. Crops needs the ann-file trim
    # info too because trimmed-empty files are also orphans worth flagging.
    png_drops, ann_rewrites, ann_deletes = _plan_crops_drain(
        repo_files, promoted_crops, api)
    crops_orphans = len(png_drops) + len(ann_deletes) + len(ann_rewrites)

    screen_drops = _plan_screens_drain(repo_files, promoted_screens)
    screens_orphans = len(screen_drops)

    anchor_drops: list[str] = []
    if args.include_anchors:
        anchor_drops = _plan_anchors_drain(repo_files, promoted_anchors, api)

    print(f'== Knowledge repo: {HF_KNOW}')
    know_files    = _list_repo_files(api, HF_KNOW, 'dataset')
    processed_ids = _load_knowledge_processed(api)
    print(f'   {len(know_files)} paths listed, '
          f'{len(processed_ids)} processed contribution IDs.')
    contrib_drops = _plan_contributions_drain(know_files, processed_ids)
    # Each contribution is .json + .png pair — count pairs, not files.
    contribs_orphans = sum(1 for p in contrib_drops if p.endswith('.json'))

    print()
    print('─── Audit results ────────────────────────────────────────')
    breached = False
    breached |= _emit_audit_line('crops',         crops_orphans,    args.crops_max)
    breached |= _emit_audit_line('screens',       screens_orphans,  args.screens_max)
    breached |= _emit_audit_line('contributions', contribs_orphans, args.contribs_max)
    if args.include_anchors:
        breached |= _emit_audit_line('anchors',   len(anchor_drops), args.anchors_max)

    print()
    if breached:
        print('FAIL: at least one domain breached its threshold.')
        print('      Run admin_drain_stale_staging.py via the '
              'drain_stale_staging.yml workflow to clean up after '
              'verifying root cause.')
        return 1

    print('OK: all domains within thresholds.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
