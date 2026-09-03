#!/usr/bin/env python3
"""
admin_audit_pipeline_movement.py — is the pipeline actually moving?
===================================================================
Every other audit here checks a *state*: how much staging has piled up, how
many mislabelled crops are in `data/`. None of them checks whether anything
is still flowing, and that is the gap this closes.

On 2026-07-16 the crop merge began failing on every scheduled run. It kept
computing the right answer — 2281 crops ready to promote — and then died on
an opaque HTTP 400 while committing. The workflow reported success for seven
weeks, because the step piped its output through `tee` and bash returns the
exit status of the last command in a pipeline. Nobody noticed until a crop
was traced by hand for an unrelated reason.

The lesson is not the 400. Any of a dozen causes would have produced the same
silence: a bad token, a raised vote threshold, a crashed merger, a renamed
path. What was missing is a statement of the obvious:

    if uploads keep arriving and `data/` never changes, something is broken

That is the whole of this check. It is deliberately blind to *why*.

Healthy steady state: `data/annotations.jsonl` changes at least every
`--max-age-days` while staging holds work. Breach exits 1 so the scheduled
workflow fails and mails the repo owner — same pattern as
`admin_audit_staging.py` and `admin_audit_virtual_poison.py`.

Read-only. Never commits.

Usage:
    .venv/bin/python admin_audit_pipeline_movement.py
    .venv/bin/python admin_audit_pipeline_movement.py --max-age-days 3

Environment (.env, same as the mergers):
    HF_TOKEN — read token (any HF token works)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

REPO = 'sets-sto/sto-icon-dataset'
RTYPE = 'dataset'
DATA_ANN = 'data/annotations.jsonl'
STAGING_PREFIX = 'staging/'

# How long `data/` may stand still while staging holds work before this is
# called a breach. The merge runs every two hours, so a day of silence is
# already dozens of failed attempts; two days leaves room for a quiet
# weekend and a slow upstream without crying wolf.
DEFAULT_MAX_AGE_DAYS = 2


def _residue(files: list[str]) -> list[str]:
    """Staging files that no run can ever settle.

    The mergers drain whatever they tally, so an entry that cannot be tallied
    is not pending — it is left over, and it accumulates silently until
    somebody remembers a cleanup script. `admin_drain_stale_staging.py` had
    not run since 2026-07-17 when this was written, and ten crops were
    sitting in staging with no annotation row to tally them.

    Reported, never deleted: what produced it matters more than the file.
    """
    from collections import defaultdict

    rows_by_iid: dict[str, set[str]] = defaultdict(set)
    ann_by_iid:  set[str]            = set()
    for f in files:
        parts = f.split('/')
        if len(parts) >= 3 and parts[0] == 'staging' and parts[-1] == 'annotations.jsonl':
            ann_by_iid.add(parts[1])

    # Reading the rows needs file contents, which this audit deliberately does
    # not download. An install with crops but no annotations file at all is
    # detectable from the listing alone, and is the shape that actually
    # occurred.
    crops_by_iid: dict[str, int] = defaultdict(int)
    for f in files:
        parts = f.split('/')
        if (len(parts) >= 4 and parts[0] == 'staging'
                and parts[2] == 'crops' and f.endswith('.png')):
            crops_by_iid[parts[1]] += 1

    out: list[str] = []
    for iid, n in sorted(crops_by_iid.items()):
        if iid not in ann_by_iid:
            out.append(f'{n} crop(s) under staging/{iid}/ with no '
                       f'annotations.jsonl — nothing can tally them')
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description='Fail when the crop pipeline stops moving.')
    ap.add_argument('--max-age-days', type=float, default=DEFAULT_MAX_AGE_DAYS,
                    help=f'Days `data/` may stand still while staging holds '
                         f'work (default {DEFAULT_MAX_AGE_DAYS}).')
    ap.add_argument('--token', default=os.environ.get('HF_TOKEN', ''))
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print('huggingface_hub not installed', file=sys.stderr)
        return 2

    api = HfApi(token=args.token or None)

    files = api.list_repo_files(REPO, repo_type=RTYPE)
    staging_crops = [f for f in files
                     if f.startswith(STAGING_PREFIX) and f.endswith('.png')]

    commits = api.list_repo_commits(REPO, repo_type=RTYPE)
    now = datetime.now(timezone.utc)

    # The promotion is the only thing that rewrites data/annotations.jsonl, so
    # its commit is the pipeline's heartbeat. Matching on the message rather
    # than on the tree keeps this to one API call and one obvious rule.
    promotions = [c for c in commits if c.title.startswith('democratic_merge:')]
    last_promotion = promotions[0].created_at if promotions else None

    print('=' * 64)
    print(f'Pipeline movement — {REPO}')
    print('=' * 64)
    print(f'staging crops waiting : {len(staging_crops)}')
    if last_promotion is None:
        print('last promotion        : never')
    else:
        age = now - last_promotion
        print(f'last promotion        : {last_promotion:%Y-%m-%d %H:%M} UTC '
              f'({age.days}d {age.seconds // 3600}h ago)')

    residue = _residue(files)
    if residue:
        print()
        for line in residue:
            print(f'  RESIDUE: {line}')

    if not staging_crops and not residue:
        print('\nOK — nothing is waiting, so a quiet `data/` is expected.')
        return 0

    if residue:
        print('\nBREACH — staging holds files no run can settle. The mergers '
              'drain what they tally, so this is left over from a failure or '
              'from a path nothing reads. It should never need a manual '
              'script; find what produced it.')
        return 1

    if last_promotion is None:
        print(f'\nBREACH — {len(staging_crops)} crops are waiting and `data/` has '
              f'never been written.')
        return 1

    limit = timedelta(days=args.max_age_days)
    if now - last_promotion > limit:
        print(f'\nBREACH — {len(staging_crops)} crops are waiting and the last '
              f'promotion was {(now - last_promotion).days} days ago '
              f'(limit {args.max_age_days}).')
        print('The merge is running and not landing. Check the Merge Staging '
              'run log for the commit step, not the summary line: the merge '
              'reports what it *would* promote before it tries to commit.')
        return 1

    print('\nOK — the pipeline is moving.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
