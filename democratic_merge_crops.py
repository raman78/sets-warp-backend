#!/usr/bin/env python3
"""
democratic_merge_crops.py — WARP crop dataset majority-vote merger
===================================================================
Companion to admin_merge.py. Operates on the OTHER HF repo:

    admin_merge.py            sets-sto/warp-knowledge      key = phash
    democratic_merge_crops.py sets-sto/sto-icon-dataset    key = crop_sha256

Reads every staging/<install_id>/annotations.jsonl, tallies votes per
crop_sha256, and publishes the winners to:
    - data/annotations.jsonl   (one line per approved sha)
    - data/crops/<sha>.png     (byte-exact crop)

Rules:
    - one vote per (install_id, sha): duplicate uploads don't stack
    - threshold: --min (default 2) for sha already in data/, 1 for new sha
    - poison filter: names starting with '__' or 'Test Item Name' dropped
      (virtual classes used as training markers — must never leak into
      the lookup table)
    - slot is metadata; the most-common slot among voters for the winning
      name is recorded
    - all writes go out in ONE HF commit so the dataset is never observed
      half-applied

Usage:
    python democratic_merge_crops.py                  # dry-run
    python democratic_merge_crops.py --apply          # commit to HF
    python democratic_merge_crops.py --apply --min 1  # 1 vote enough
    python democratic_merge_crops.py --since 2026-03-01
    python democratic_merge_crops.py --verbose

Environment variables (or .env file — same as admin_merge.py):
    HF_TOKEN     — HF write token (required)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
UTC = timezone.utc
from pathlib import Path


# ── .env loader (mirrors admin_merge.py) ──────────────────────────────────

def _load_env():
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()

HF_TOKEN = os.environ.get('HF_TOKEN', '')


REPO  = 'sets-sto/sto-icon-dataset'
RTYPE = 'dataset'

DATA_ANN = 'data/annotations.jsonl'
DATA_CRP = 'data/crops'


def _is_poison_name(name: str) -> bool:
    """
    Names that must NEVER enter data/annotations.jsonl:
      - virtual classes (__empty__, __inactive__, __boff_*) — would hard-
        override real icons to 'empty/inactive' with conf=1.0 at inference.
      - leftover dev-test entries.
    """
    return name.startswith('__') or name == 'Test Item Name'


def _load_existing(api, token: str) -> dict[str, dict]:
    """Load current data/annotations.jsonl → dict keyed by crop_sha256."""
    from huggingface_hub import hf_hub_download
    out: dict[str, dict] = {}
    try:
        local = hf_hub_download(
            repo_id=REPO, filename=DATA_ANN, repo_type=RTYPE, token=token)
    except Exception as e:
        print(f'NOTICE: data/annotations.jsonl does not exist ({e}) '
              f'— starting from scratch')
        return out
    with open(local, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d   = json.loads(line)
                sha = d.get('crop_sha256')
                if sha:
                    out[sha] = d
            except Exception:
                pass
    return out


def _collect_votes(
    api, token: str, since: str | None, repo_files: set[str],
) -> tuple[dict[str, Counter], dict[str, Counter], dict[str, str], dict[str, int]]:
    """
    Fetch all staging annotations and tally votes.

    Returns (name_votes, slot_votes, crop_src, per_install):
      - name_votes[sha][name]  → count of distinct install_ids voting for name
      - slot_votes[sha][slot]  → same, for slot
      - crop_src[sha]          → first staging path that has this crop PNG
      - per_install[install_id]→ number of entries contributed by that install
    """
    from hf_clone import clone_hf_shallow

    print('Cloning staging tree (shallow, sparse)…')
    # Sparse-checkout staging/ only — repo also contains data/crops/*.png
    # which this function does not need (those are fetched on demand later
    # via hf_hub_download).
    snap_dir = clone_hf_shallow(REPO, token, repo_type=RTYPE, paths=['staging'])
    root = Path(snap_dir) / 'staging'
    if not root.exists():
        print(f'WARNING: no staging/ folder at {root}')
        return {}, {}, {}, {}

    anno_files = sorted(root.glob('*/annotations.jsonl'))
    print(f'Found {len(anno_files)} contributors with annotations.')

    name_votes: dict[str, Counter] = defaultdict(Counter)
    slot_votes: dict[str, Counter] = defaultdict(Counter)
    crop_src:   dict[str, str]     = {}
    per_install: dict[str, int]    = {}

    for f in repo_files:
        if f.startswith('staging/') and f.endswith('.png'):
            crop_src.setdefault(Path(f).stem, f)

    for af in anno_files:
        install_id = af.parent.name

        seen_in_install: set[tuple[str, str, str]] = set()
        n_entries = 0
        try:
            with open(af, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    sha  = (e.get('crop_sha256') or '').strip()
                    name = (e.get('name') or '').strip()
                    slot = (e.get('slot') or '').strip()
                    if not sha or not name:
                        continue
                    if since:
                        date = (e.get('date') or '').strip()
                        if date and date < since:
                            continue
                    if _is_poison_name(name):
                        continue
                    # Dedup duplicate uploads from one install: one vote each.
                    key = (sha, name, slot)
                    if key in seen_in_install:
                        continue
                    seen_in_install.add(key)
                    name_votes[sha][name] += 1
                    if slot:
                        slot_votes[sha][slot] += 1
                    n_entries += 1
        except Exception as e:
            print(f'  SKIP {install_id}: {e}')
            continue
        per_install[install_id] = n_entries

    return name_votes, slot_votes, crop_src, per_install


def _merge(
    name_votes: dict[str, Counter],
    slot_votes: dict[str, Counter],
    existing:   dict[str, dict],
    min_votes:  int,
    verbose:    bool,
) -> tuple[dict[str, dict], list[dict]]:
    """Majority vote. Returns (merged, report_rows)."""
    # Drop legacy poison entries from existing (one-shot self-heal).
    merged = {sha: rec for sha, rec in existing.items()
              if not _is_poison_name((rec.get('name') or ''))}
    dropped_poison = len(existing) - len(merged)
    if dropped_poison:
        print(f'[clean] dropped {dropped_poison} legacy poison entries')

    report: list[dict] = []

    for sha, votes in sorted(name_votes.items()):
        winner, count = votes.most_common(1)[0]
        old_rec       = existing.get(sha)
        old_name      = (old_rec or {}).get('name', '')

        threshold = min_votes if sha in existing else 1
        accepted  = count >= threshold

        action = 'SKIP'
        if accepted:
            if old_name == winner:
                action = 'unchanged'
            elif old_name:
                action = 'UPDATE'
            else:
                action = 'NEW'

            slot_c = slot_votes.get(sha) or Counter()
            slot   = slot_c.most_common(1)[0][0] if slot_c else ''
            merged[sha] = {
                'crop_sha256': sha,
                'name':        winner,
                'slot':        slot,
                'votes':       count,
                'updated_at':  datetime.now(UTC).isoformat(timespec='seconds')
                                                .replace('+00:00', 'Z'),
            }

        row = {
            'sha':      sha,
            'winner':   winner,
            'votes':    count,
            'total':    sum(votes.values()),
            'old_name': old_name,
            'action':   action,
        }
        # losers worth surfacing
        losers = [(n, v) for n, v in votes.most_common() if n != winner]
        if losers:
            row['losers'] = dict(losers[:3])
        report.append(row)

        if verbose or action in ('NEW', 'UPDATE', 'SKIP'):
            _print_row(row)

    return merged, report


def _print_row(row: dict):
    action = row['action']
    symbol = {'NEW': '+', 'UPDATE': '~', 'unchanged': '.', 'SKIP': '-'}.get(action, '?')
    sha    = row['sha'][:12]
    winner = row['winner'][:42]
    votes  = row['votes']
    total  = row['total']
    old    = f"  (was: {row['old_name'][:30]})" if row.get('old_name') and action == 'UPDATE' else ''
    losers = f"  losers={row.get('losers')}" if row.get('losers') else ''
    print(f'  {symbol} [{sha}] {winner!r:44s} {votes}/{total}{old}{losers}')


def _apply(
    api, token: str,
    merged:   dict[str, dict],
    existing: dict[str, dict],
    crop_src: dict[str, str],
    repo_files: set[str],
):
    """One commit: rewrite data/annotations.jsonl + add any missing crops."""
    from huggingface_hub import (
        CommitOperationAdd, hf_hub_download,
    )

    # 1. Annotations file (one JSON object per line, sorted by sha for diffability).
    lines: list[str] = []
    for sha in sorted(merged):
        lines.append(json.dumps(merged[sha], ensure_ascii=False))
    payload = ('\n'.join(lines) + '\n').encode('utf-8')

    ops: list = [
        CommitOperationAdd(
            path_in_repo  = DATA_ANN,
            path_or_fileobj = io.BytesIO(payload),
        )
    ]

    # 2. Copy any approved crop that isn't already in data/crops/.
    missing: list[str] = []
    for sha in merged:
        dst = f'{DATA_CRP}/{sha}.png'
        if dst in repo_files:
            continue
        src = crop_src.get(sha)
        if not src:
            missing.append(sha)
            continue
        try:
            local = hf_hub_download(
                repo_id=REPO, filename=src, repo_type=RTYPE, token=token)
        except Exception as e:
            print(f'  ERR fetching {src}: {e}')
            missing.append(sha)
            continue
        ops.append(CommitOperationAdd(
            path_in_repo    = dst,
            path_or_fileobj = local,
        ))

    if missing:
        print(f'WARNING: {len(missing)} approved sha have no fetchable crop '
              f'— annotations will reference missing files.')

    print(f'Committing: 1 annotations file + {len(ops) - 1} new crops…')
    api.create_commit(
        repo_id        = REPO,
        repo_type      = RTYPE,
        operations     = ops,
        commit_message = (f'democratic_merge: {len(merged)} entries '
                          f'({len(ops) - 1} new crops) '
                          f'@ {datetime.now(UTC).strftime("%Y-%m-%d %H:%M")} UTC'),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description='Democratic majority-vote merger for the crop dataset.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--token',   default=HF_TOKEN,
                    help='HF write token (falls back to $HF_TOKEN / .env)')
    ap.add_argument('--apply',   action='store_true',
                    help='Commit to HF (default: dry-run)')
    ap.add_argument('--min',     type=int, default=2, metavar='N',
                    help='Minimum votes for existing sha (default: 2). '
                         'New sha always require only 1.')
    ap.add_argument('--since',   metavar='YYYY-MM-DD',
                    help='Only count entries with date >= this')
    ap.add_argument('--verbose', action='store_true',
                    help='Print every sha, not only changes')
    ap.add_argument('--export',  metavar='FILE',
                    help='Also save resulting annotations.jsonl locally')
    args = ap.parse_args()

    if not args.token:
        print('ERROR: HF_TOKEN not set (env, .env, or --token)', file=sys.stderr)
        return 1

    print('=' * 64)
    print(f'Democratic merger — {REPO}')
    print(f'Min votes: {args.min}  ·  Mode: {"APPLY" if args.apply else "DRY-RUN"}'
          + (f'  ·  Since: {args.since}' if args.since else ''))
    print('=' * 64)

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)

    print('Listing repo files…')
    repo_files = set(api.list_repo_files(repo_id=REPO, repo_type=RTYPE))

    existing = _load_existing(api, args.token)
    print(f'Existing data/annotations.jsonl: {len(existing)} entries')

    name_votes, slot_votes, crop_src, per_install = _collect_votes(
        api, args.token, since=args.since, repo_files=repo_files)
    print(f'Contributors: {len(per_install)}   '
          f'unique sha hashes voted on: {len(name_votes)}')

    if not name_votes:
        print('No staging entries to merge — nothing to do.')
        return 0

    merged, report = _merge(
        name_votes, slot_votes, existing,
        min_votes=args.min, verbose=args.verbose)

    new_count    = sum(1 for r in report if r['action'] == 'NEW')
    update_count = sum(1 for r in report if r['action'] == 'UPDATE')
    skip_count   = sum(1 for r in report if r['action'] == 'SKIP')
    unchanged    = sum(1 for r in report if r['action'] == 'unchanged')

    print()
    print('─── Summary ──────────────────────────────────────────')
    print(f'  + NEW:       {new_count}')
    print(f'  ~ UPDATE:    {update_count}')
    print(f'  . unchanged: {unchanged}')
    print(f'  - SKIP:      {skip_count}  (below threshold)')
    print(f'  Total after merge: {len(merged)} entries')

    if args.export:
        Path(args.export).write_text(
            '\n'.join(json.dumps(merged[s], ensure_ascii=False)
                      for s in sorted(merged)) + '\n',
            encoding='utf-8')
        print(f'Local export → {args.export}')

    if not args.apply:
        print('\nDRY-RUN — use --apply to commit.')
        return 0

    _apply(api, args.token, merged, existing, crop_src, repo_files)
    print('OK — committed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
