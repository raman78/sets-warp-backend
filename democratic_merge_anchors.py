#!/usr/bin/env python3
"""
democratic_merge_anchors.py — anchor-grid median aggregation merger
====================================================================
Companion to admin_merge.py / democratic_merge_crops.py. Operates on
the icon dataset:

    sets-sto/sto-icon-dataset
        staging/<install_id>/anchors_grid_<sha8>.json   (per-grid inputs)
        data/anchors/<build_type>_<aspect_bucket>.json  (consensus output)

Per the audit (D-F.1 / D-F.2 / D-F.4 / D-F.5):
    - Groups grids by (build_type, aspect_bucket=round(aspect, 2)).
    - One vote per install_id (duplicate uploads from a single install
      contribute one set of slot coordinates each, but they all roll into
      the median together — there is no extra dedup beyond unique-iid
      grouping).
    - Threshold: NEW group needs >=1 contributor, UPDATE of an existing
      data/anchors/<key>.json needs >=2 (asymmetric, matches D-A.4).
    - Per-slot aggregation: median over every contributing grid's coord
      (x0_rel, y_rel, w_rel, h_rel, step_rel, count). Slots that fewer
      than the threshold of grids carried are dropped — we never emit a
      slot with too thin support.
    - `spread` (min / max / stddev) is recorded per slot per coord as the
      audit-trail surrogate for "losers" (anchors don't have discrete
      losing options, only outliers).
    - After commit, every staging/<iid>/anchors_grid_<sha8>.json whose
      content was folded into a promoted group is DELETED — same atomic
      commit as the data/anchors/ write.

This script replaces `build_community_anchors` / `upload_community_anchors`
in admin_train.py (to be removed in PHASE 3 per D-F.2). The trainer will
read data/anchors/ directly instead of voting on its own.

Usage:
    python democratic_merge_anchors.py                 # dry-run
    python democratic_merge_anchors.py --apply         # commit to HF
    python democratic_merge_anchors.py --apply --min 1 # 1 contributor OK
                                                       # even for UPDATEs

Environment variables (same as the other mergers):
    HF_TOKEN     — HF write token (required)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
UTC = timezone.utc
from pathlib import Path


# ── .env loader (mirrors admin_merge.py / democratic_merge_crops.py) ──────

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

DATA_DIR = 'data/anchors'

# Coordinate keys we median-aggregate. `count` is integer-rounded, the
# rest float-rounded to 5 decimals (matches the original
# build_community_anchors precision).
_COORD_KEYS = ('x0_rel', 'y_rel', 'w_rel', 'h_rel', 'step_rel')


def _grid_key(build_type: str, aspect: float | None) -> tuple[str, float] | None:
    """Return the (build_type, aspect_bucket) grouping key, or None when
    inputs are malformed (no build_type / no aspect)."""
    if not build_type or aspect is None:
        return None
    try:
        bucket = round(float(aspect), 2)
    except (TypeError, ValueError):
        return None
    return (build_type, bucket)


def _key_to_path(key: tuple[str, float]) -> str:
    """Filename for the consensus artefact — embed bucket in the name so
    a single build_type can have multiple aspect groups side-by-side."""
    build_type, bucket = key
    # `_` separator chosen so the file is grep-friendly. `.2f` keeps the
    # bucket exactly 2 decimals (matches round(aspect, 2) at parse time).
    return f'{DATA_DIR}/{build_type}_{bucket:.2f}.json'


def _load_existing(token: str) -> dict[tuple[str, float], dict]:
    """List every data/anchors/<...>.json file in HF and parse them."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
    api = HfApi(token=token)
    out: dict[tuple[str, float], dict] = {}
    try:
        for f in api.list_repo_files(repo_id=REPO, repo_type=RTYPE):
            if not f.startswith(DATA_DIR + '/') or not f.endswith('.json'):
                continue
            try:
                local = hf_hub_download(
                    repo_id=REPO, filename=f, repo_type=RTYPE, token=token)
                rec   = json.loads(Path(local).read_text(encoding='utf-8'))
                bt    = (rec.get('build_type') or '').strip()
                ab    = rec.get('aspect_bucket')
                if bt and ab is not None:
                    out[(bt, round(float(ab), 2))] = rec
            except Exception as e:
                print(f'  SKIP existing {f}: {e}')
    except HfHubHTTPError as e:
        # First-run: data/anchors/ does not exist yet — start clean.
        print(f'NOTICE: data/anchors/ not listable ({e}) — starting from scratch')
    return out


def _collect_votes(
    token: str,
) -> tuple[
    dict[tuple[str, float], dict[str, list[dict]]],
    dict[tuple[str, float], list[str]],
    dict[tuple[str, float], list[str]],
    dict[str, list[tuple[tuple[str, float], str]]],
]:
    """
    Clone staging shallowly and tally per-key contributions.

    Returns (groups, group_resolutions, group_iids, staging_index):
      - groups[key][install_id] → list of grids (one per file) from that
        install for that (build_type, aspect_bucket).
      - group_resolutions[key]  → flat list of every "WIDTHxHEIGHT" seen
        (used to surface the most common resolution in the artefact).
      - group_iids[key]         → ordered list of unique install_ids that
        contributed to this group (preserves contribution order).
      - staging_index[install_id] → [(key, filename), ...] — used by the
        drain to map a promoted key back to the on-disk grid files.
    """
    from hf_clone import clone_hf_shallow

    print('Cloning staging tree (shallow)…')
    snap_dir = clone_hf_shallow(REPO, token, repo_type=RTYPE)
    root = Path(snap_dir) / 'staging'
    if not root.exists():
        print(f'WARNING: no staging/ folder at {root}')
        return {}, {}, {}, {}

    grid_files = sorted(root.glob('*/anchors_grid_*.json'))
    print(f'Found {len(grid_files)} anchor grid file(s) across '
          f'{len(set(p.parent.name for p in grid_files))} contributor(s).')

    groups:        dict[tuple[str, float], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    group_res:     dict[tuple[str, float], list[str]] = defaultdict(list)
    group_iids:    dict[tuple[str, float], list[str]] = defaultdict(list)
    staging_index: dict[str, list[tuple[tuple[str, float], str]]] = defaultdict(list)

    for gp in grid_files:
        install_id = gp.parent.name
        try:
            entry = json.loads(gp.read_text(encoding='utf-8'))
        except Exception as e:
            print(f'  SKIP {install_id}/{gp.name}: {e}')
            continue
        key = _grid_key(entry.get('build_type', ''), entry.get('aspect'))
        if key is None:
            continue
        groups[key][install_id].append(entry)
        staging_index[install_id].append((key, gp.name))
        if install_id not in group_iids[key]:
            group_iids[key].append(install_id)
        res = (entry.get('resolution') or '').strip()
        if res:
            group_res[key].append(res)

    return groups, group_res, group_iids, staging_index


def _aggregate_group(
    contributors: dict[str, list[dict]],
    resolutions:  list[str],
    iids:         list[str],
) -> dict | None:
    """Median-aggregate one (build_type, aspect_bucket) group.

    Returns the artefact body, or None when no slot survives the
    threshold (e.g. every grid had only 0–1 slot).
    """
    # All raw grids from all contributors, flattened.
    grids: list[dict] = [g for grids in contributors.values() for g in grids]

    # Per-slot list of bbox dicts, one per grid that defined that slot.
    slot_geos: dict[str, list[dict]] = defaultdict(list)
    for g in grids:
        for slot, geo in (g.get('slots') or {}).items():
            if isinstance(geo, dict):
                slot_geos[slot].append(geo)

    aggregated: dict[str, dict] = {}
    spread:     dict[str, dict] = {}
    for slot, geos in slot_geos.items():
        # Skip slots that fewer than one grid carried (defensive — should
        # not happen because the slot only appears if any grid had it).
        if not geos:
            continue
        entry: dict = {}
        slot_spread: dict = {}
        for key in _COORD_KEYS:
            vals = [float(g[key]) for g in geos
                    if key in g and isinstance(g[key], (int, float))]
            if not vals:
                continue
            entry[key] = round(statistics.median(vals), 5)
            slot_spread[key] = {
                'min':    round(min(vals), 5),
                'max':    round(max(vals), 5),
                'stddev': round(statistics.pstdev(vals), 5) if len(vals) > 1 else 0.0,
                'n':      len(vals),
            }
        counts = [int(g['count']) for g in geos
                  if 'count' in g and isinstance(g['count'], (int, float))]
        if counts:
            entry['count'] = int(round(statistics.median(counts)))
        # A slot is only emitted when the four mandatory coords survived
        # — otherwise the median dropped one and the slot is unusable.
        if all(k in entry for k in ('x0_rel', 'y_rel', 'w_rel', 'h_rel')):
            aggregated[slot] = entry
            spread[slot]     = slot_spread

    if not aggregated:
        return None

    rep_res = ''
    if resolutions:
        counts = Counter(resolutions)
        rep_res = counts.most_common(1)[0][0]

    return {
        'slots':           aggregated,
        'spread':          spread,
        'n_contributors':  len(iids),
        'install_ids':     list(iids),
        'resolutions':     dict(Counter(resolutions)),
        'representative_resolution': rep_res,
    }


def _merge(
    groups:       dict[tuple[str, float], dict[str, list[dict]]],
    group_res:    dict[tuple[str, float], list[str]],
    group_iids:   dict[tuple[str, float], list[str]],
    existing:     dict[tuple[str, float], dict],
    min_votes:    int,
    verbose:      bool,
) -> tuple[dict[tuple[str, float], dict], list[dict], set[tuple[str, float]]]:
    """Median-aggregate per group. Returns (merged, report, promoted_keys).

    `promoted_keys` is the set of (build_type, aspect_bucket) keys that
    cleared their NEW=1 / UPDATE>=min_votes threshold and were folded
    into the merged artefact. The drain in `_apply` uses it to know
    which staging files to delete.
    """
    merged:  dict[tuple[str, float], dict] = dict(existing)
    report:  list[dict] = []
    promoted: set[tuple[str, float]] = set()

    for key in sorted(groups):
        contributors = groups[key]
        n_iids       = len(group_iids[key])
        is_update    = key in existing
        threshold    = min_votes if is_update else 1

        action = 'SKIP'
        if n_iids >= threshold:
            body = _aggregate_group(contributors, group_res[key], group_iids[key])
            if body is None:
                action = 'SKIP'
            else:
                body['build_type']    = key[0]
                body['aspect_bucket'] = key[1]
                body['updated_at']    = datetime.now(UTC).isoformat(timespec='seconds')\
                                                        .replace('+00:00', 'Z')
                if is_update and merged[key].get('slots') == body.get('slots'):
                    action = 'unchanged'
                elif is_update:
                    action = 'UPDATE'
                else:
                    action = 'NEW'
                merged[key] = body
                promoted.add(key)

        row = {
            'key':         f'{key[0]}@{key[1]:.2f}',
            'contributors':n_iids,
            'threshold':   threshold,
            'action':      action,
            'slots':       len(merged.get(key, {}).get('slots') or {})
                                if action != 'SKIP' else 0,
        }
        report.append(row)
        if verbose or action in ('NEW', 'UPDATE', 'SKIP'):
            _print_row(row)

    return merged, report, promoted


def _print_row(row: dict):
    symbol = {'NEW': '+', 'UPDATE': '~', 'unchanged': '.', 'SKIP': '-'}.get(row['action'], '?')
    print(f'  {symbol} {row["key"]:30s} '
          f'iids={row["contributors"]:<3} thr={row["threshold"]} '
          f'slots={row["slots"]:<3} action={row["action"]}')


def _apply(
    api, token: str,
    merged:        dict[tuple[str, float], dict],
    promoted:      set[tuple[str, float]],
    staging_index: dict[str, list[tuple[tuple[str, float], str]]],
):
    """Single atomic commit: write data/anchors/<key>.json for each
    promoted group AND delete every staging anchor grid file whose
    content rolled into a promoted group."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    ops: list = []
    for key in sorted(promoted):
        body = merged[key]
        path = _key_to_path(key)
        ops.append(CommitOperationAdd(
            path_in_repo    = path,
            path_or_fileobj = io.BytesIO(
                json.dumps(body, ensure_ascii=False, indent=2).encode('utf-8')
            ),
        ))

    drained = 0
    for iid, items in staging_index.items():
        for key, fname in items:
            if key in promoted:
                ops.append(CommitOperationDelete(
                    path_in_repo=f'staging/{iid}/{fname}',
                ))
                drained += 1

    if not ops:
        print('No ops to commit.')
        return

    print(f'Committing: {len(promoted)} anchor groups + drain({drained} staging grids)…')
    api.create_commit(
        repo_id        = REPO,
        repo_type      = RTYPE,
        operations     = ops,
        commit_message = (f'democratic_merge_anchors: {len(promoted)} groups '
                          f'(drained {drained} staging grids) '
                          f'@ {datetime.now(UTC).strftime("%Y-%m-%d %H:%M")} UTC'),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description='Democratic median-aggregation merger for anchor grids.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--token',   default=HF_TOKEN,
                    help='HF write token (falls back to $HF_TOKEN / .env)')
    ap.add_argument('--apply',   action='store_true',
                    help='Commit to HF (default: dry-run)')
    ap.add_argument('--min',     type=int, default=2, metavar='N',
                    help='Minimum contributors to UPDATE an existing group '
                         '(default: 2). NEW groups always need only 1.')
    ap.add_argument('--verbose', action='store_true',
                    help='Print every group, not only changes')
    args = ap.parse_args()

    if not args.token:
        print('ERROR: HF_TOKEN not set (env, .env, or --token)', file=sys.stderr)
        return 1

    print('=' * 64)
    print(f'Democratic anchor merger — {REPO}')
    print(f'Min votes: {args.min}  ·  Mode: {"APPLY" if args.apply else "DRY-RUN"}')
    print('=' * 64)

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError
    api = HfApi(token=args.token)

    existing = _load_existing(args.token)
    print(f'Existing data/anchors/: {len(existing)} consensus group(s)')

    # HF tree-list is rate-limited; retry 429 with exponential backoff
    # mirrors the pattern in democratic_merge_crops.py.
    for attempt in range(5):
        try:
            groups, group_res, group_iids, staging_index = _collect_votes(args.token)
            break
        except HfHubHTTPError as e:
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            if status != 429 or attempt == 4:
                raise
            delay = 2 ** attempt
            print(f'  HF 429 on _collect_votes (attempt {attempt + 1}/5) — '
                  f'sleeping {delay}s', file=sys.stderr)
            time.sleep(delay)

    if not groups:
        print('No staging anchor grids — nothing to do.')
        return 0

    merged, report, promoted = _merge(
        groups, group_res, group_iids, existing,
        min_votes=args.min, verbose=args.verbose,
    )

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
    print(f'  Total groups after merge: {len(merged)}')

    if not args.apply:
        print('\nDRY-RUN — use --apply to commit.')
        return 0

    if not promoted:
        print('Nothing promoted — no commit needed.')
        return 0

    _apply(api, args.token, merged, promoted, staging_index)
    print('OK — committed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
