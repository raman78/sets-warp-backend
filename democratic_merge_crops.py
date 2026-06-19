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
import subprocess
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
      - internal test markers (__boff_*, Test Item Name)
      - leftover dev-test entries.

    __empty__ and __inactive__ are ALLOWED — the ArcFace embedder needs
    them as gallery classes so inactive/empty slots match to their own
    class instead of nearest-neighbour-snapping to a real ability.
    The pHash override path in icon_matcher already suppresses virtual
    names independently (name.startswith('__') → suppress=True).
    """
    if name in ('__empty__', '__inactive__'):
        return False
    return name.startswith('__') or name == 'Test Item Name'


def _load_existing(snap_dir) -> dict[str, dict]:
    """Load current data/annotations.jsonl → dict keyed by crop_sha256.

    Reads from the local shallow clone — avoids an extra HF resolve request.
    """
    out: dict[str, dict] = {}
    local = Path(snap_dir) / DATA_ANN
    if not local.exists():
        print(f'NOTICE: {DATA_ANN} not in clone — starting from scratch')
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
    snap_dir, since: str | None, repo_files: set[str],
) -> tuple[
    dict[str, Counter],
    dict[str, Counter],
    dict[str, str],
    dict[str, int],
    dict[str, set[str]],
    dict[str, list[dict]],
]:
    """
    Tally votes from staging annotations in the local shallow clone.

    Returns (name_votes, slot_votes, crop_src, per_install,
             contributors_for_sha, staging_records):
      - name_votes[sha][name]   → count of distinct install_ids voting for name
      - slot_votes[sha][slot]   → same, for slot
      - crop_src[sha]           → first staging path that has this crop PNG
      - per_install[install_id] → number of entries contributed by that install
      - contributors_for_sha[sha] → install_ids whose annotations voted on sha
        (used by drain — delete staging/<iid>/crops/<sha>.png after promotion)
      - staging_records[install_id] → raw annotation dicts kept for the
        staging rewrite (drain trims entries whose sha was promoted)
    """
    root = Path(snap_dir) / 'staging'
    if not root.exists():
        print(f'WARNING: no staging/ folder at {root}')
        return {}, {}, {}, {}, {}, {}

    anno_files = sorted(root.glob('*/annotations.jsonl'))
    print(f'Found {len(anno_files)} contributors with annotations.')

    name_votes: dict[str, Counter] = defaultdict(Counter)
    slot_votes: dict[str, Counter] = defaultdict(Counter)
    crop_src:   dict[str, str]     = {}
    per_install: dict[str, int]    = {}
    contributors_for_sha: dict[str, set[str]] = defaultdict(set)
    staging_records:      dict[str, list[dict]] = {}

    for f in repo_files:
        if f.startswith('staging/') and f.endswith('.png'):
            crop_src.setdefault(Path(f).stem, f)

    for af in anno_files:
        install_id = af.parent.name

        seen_in_install: set[tuple[str, str, str]] = set()
        records: list[dict] = []
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
                    records.append(e)
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
                    contributors_for_sha[sha].add(install_id)
                    n_entries += 1
        except Exception as e:
            print(f'  SKIP {install_id}: {e}')
            continue
        per_install[install_id] = n_entries
        staging_records[install_id] = records

    return (name_votes, slot_votes, crop_src, per_install,
            contributors_for_sha, staging_records)


def _merge(
    name_votes: dict[str, Counter],
    slot_votes: dict[str, Counter],
    existing:   dict[str, dict],
    min_votes:  int,
    verbose:    bool,
) -> tuple[dict[str, dict], list[dict], set[str]]:
    """Majority vote. Returns (merged, report_rows, promoted_shas).

    `promoted_shas` is the set this run actually accepted (NEW + UPDATE +
    unchanged). The drain in `_apply` uses this set so staging is cleaned
    even when the consensus was already reflected in data/ — those votes
    have done their job and should not be re-tallied next run.
    """
    # Drop legacy poison entries from existing (one-shot self-heal).
    merged = {sha: rec for sha, rec in existing.items()
              if not _is_poison_name((rec.get('name') or ''))}
    dropped_poison = len(existing) - len(merged)
    if dropped_poison:
        print(f'[clean] dropped {dropped_poison} legacy poison entries')

    report: list[dict] = []
    promoted_shas: set[str] = set()

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
            losers_dict = {n: v for n, v in votes.most_common()[1:4] if n != winner}
            entry: dict = {
                'schema_version': 2,
                'crop_sha256': sha,
                'name':        winner,
                'slot':        slot,
                'votes':       count,
                'updated_at':  datetime.now(UTC).isoformat(timespec='seconds')
                                                .replace('+00:00', 'Z'),
            }
            if losers_dict:
                # D-A.3: persisted dissent — minority votes preserved in
                # data/ so downstream audits can see consensus strength.
                entry['losers'] = losers_dict
            merged[sha] = entry
            promoted_shas.add(sha)

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

    return merged, report, promoted_shas


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
    merged:    dict[str, dict],
    promoted_shas: set[str],
    existing:  dict[str, dict],
    crop_src:  dict[str, str],
    repo_files: set[str],
    contributors_for_sha: dict[str, set[str]],
    staging_records: dict[str, list[dict]],
):
    """One commit: rewrite data/annotations.jsonl, copy approved crops, then
    drain staging — delete staging crop PNGs for promoted sha and rewrite
    each contributor's annotations.jsonl keeping only the not-promoted lines.

    A single atomic commit so a half-applied state is impossible.
    """
    from huggingface_hub import (
        CommitOperationAdd, CommitOperationDelete, hf_hub_download,
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
    new_crops = 0
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
        new_crops += 1

    if missing:
        print(f'WARNING: {len(missing)} approved sha have no fetchable crop '
              f'— annotations will reference missing files.')

    # 3. D-A.3 drain — delete staging crop PNGs for promoted sha, and trim
    # each contributor's annotations.jsonl. Skip sha that failed to copy
    # (`missing`) so we never delete the only remaining copy of an
    # annotated crop.
    safe_promoted = promoted_shas - set(missing)
    deleted_crops = 0
    deleted_annos = 0
    rewritten_annos = 0

    for sha in safe_promoted:
        for iid in contributors_for_sha.get(sha, ()):
            staging_png = f'staging/{iid}/crops/{sha}.png'
            if staging_png in repo_files:
                ops.append(CommitOperationDelete(path_in_repo=staging_png))
                deleted_crops += 1

    for iid, records in staging_records.items():
        kept = [r for r in records
                if (r.get('crop_sha256') or '').strip() not in safe_promoted]
        if len(kept) == len(records):
            continue
        staging_ann = f'staging/{iid}/annotations.jsonl'
        if not kept:
            ops.append(CommitOperationDelete(path_in_repo=staging_ann))
            deleted_annos += 1
        else:
            buf = ('\n'.join(json.dumps(r, ensure_ascii=False) for r in kept)
                   + '\n').encode('utf-8')
            ops.append(CommitOperationAdd(
                path_in_repo    = staging_ann,
                path_or_fileobj = io.BytesIO(buf),
            ))
            rewritten_annos += 1

    print(f'Committing: 1 annotations file + {new_crops} new crops + '
          f'drain({deleted_crops} stg crops, {rewritten_annos} stg ann '
          f'trimmed, {deleted_annos} stg ann emptied)…')
    api.create_commit(
        repo_id        = REPO,
        repo_type      = RTYPE,
        operations     = ops,
        commit_message = (f'democratic_merge: {len(merged)} entries '
                          f'(+{new_crops} new crops, drained '
                          f'{deleted_crops} staging crops) '
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

    # One shallow clone gives us BOTH the staging tree AND the list of
    # tracked files — no recursive `tree?recursive=true` API call (the
    # single biggest source of HF 429s for this script). See hf_clone.py.
    print('Cloning repo (shallow)…')
    from hf_clone import clone_hf_shallow
    snap_dir = clone_hf_shallow(REPO, args.token, repo_type=RTYPE)
    repo_files = set(
        subprocess.check_output(
            ['git', 'ls-files'], cwd=str(snap_dir), text=True,
        ).splitlines()
    )
    print(f'Repo tracks {len(repo_files)} files.')

    existing = _load_existing(snap_dir)
    print(f'Existing data/annotations.jsonl: {len(existing)} entries')

    (name_votes, slot_votes, crop_src, per_install,
     contributors_for_sha, staging_records) = _collect_votes(
        snap_dir, since=args.since, repo_files=repo_files)
    print(f'Contributors: {len(per_install)}   '
          f'unique sha hashes voted on: {len(name_votes)}')

    if not name_votes:
        print('No staging entries to merge — nothing to do.')
        return 0

    merged, report, promoted_shas = _merge(
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

    # Uniform drain monitor (post-audit TODO #5). Promoted shas equal the
    # staging crops that will be deleted next (dry-run reports the would-be
    # count; --apply reports what was actually committed).
    print(f'DRAIN: domain=crops promoted={len(promoted_shas)} '
          f'new={new_count} update={update_count} skip={skip_count}')

    if not args.apply:
        print('\nDRY-RUN — use --apply to commit.')
        return 0

    _apply(api, args.token, merged, promoted_shas, existing, crop_src,
           repo_files, contributors_for_sha, staging_records)
    print('OK — committed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
