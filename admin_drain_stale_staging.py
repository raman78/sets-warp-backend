#!/usr/bin/env python3
"""
admin_drain_stale_staging.py — One-shot drain of historic staging (post-audit TODO #3)
========================================================================================

Cleans staging artefacts that were already promoted to `data/` by the
legacy (pre-Z2) `admin_merge.py` but never deleted from the staging
queue. From the audit (`docs/data_source_audit.md` §H.8 #3):

    "Migracja istniejącego stagingu — po wdrożeniu PHASE 2 + PHASE 4,
     jeden-shot skrypt admin_drain_stale_staging.py żeby drenować
     historyczny staging (gromadzony od 2025 bez drain)."

Going forward, the 4 democratic mergers drain on promotion. This script
catches up the backlog accumulated before that change shipped.

Drain rule (content-addressed, idempotent):
    crops          — staging/<iid>/crops/<sha>.png        DROP if sha in data/annotations.jsonl
                     staging/<iid>/annotations.jsonl      TRIM lines whose sha is promoted
    screens        — staging/<iid>/screen_types/<T>/*.png DROP if sha in data/screen_types/metadata.jsonl
    contributions  — contributions/<date>/<id>.json       DROP if id in knowledge.json::processed_contributions
                     contributions/<date>/<id>.png        DROP companion crop
    anchors        — staging/<iid>/anchors_grid_*.json    OPT-IN (--include-anchors) — staging anchor
                     files aggregate multiple votes; only drain when explicitly requested
                     and the (build_type, aspect_bucket) already has a consensus file.

Two atomic commits — one per HF repo — so a half-applied state is impossible.

Usage:
    .venv/bin/python admin_drain_stale_staging.py             # dry-run
    .venv/bin/python admin_drain_stale_staging.py --apply     # commit deletes
    .venv/bin/python admin_drain_stale_staging.py --apply --include-anchors

Environment (.env, same as the mergers):
    HF_TOKEN     — write token (required)
    HF_REPO_ID   — default: sets-sto/warp-knowledge
"""

from __future__ import annotations

from hf_commit import commit_adds_then_deletes

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc


def _load_env():
    env_path = Path(__file__).parent / '.env'
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_env()

HF_TOKEN    = os.environ.get('HF_TOKEN', '')
HF_DATASET  = 'sets-sto/sto-icon-dataset'
HF_KNOW     = os.environ.get('HF_REPO_ID', 'sets-sto/warp-knowledge')


def _list_repo_files(api, repo_id: str, repo_type: str) -> list[str]:
    """Flat list of all paths in repo."""
    return list(api.list_repo_files(repo_id=repo_id, repo_type=repo_type))


def _load_jsonl_field(api, repo_id: str, path: str, key: str) -> set[str]:
    """Download a JSONL artefact and collect the set of `key` values."""
    from huggingface_hub import hf_hub_download
    try:
        local = hf_hub_download(
            repo_id=repo_id, filename=path,
            repo_type='dataset', token=HF_TOKEN,
        )
    except Exception as e:
        print(f'  {path} unavailable ({e}) — treating as empty.')
        return set()

    out: set[str] = set()
    for line in Path(local).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        v = (rec.get(key) or '').strip()
        if v:
            out.add(v)
    return out


def _load_knowledge_processed(api) -> set[str]:
    """Read knowledge.json::processed_contributions as a set of IDs."""
    from huggingface_hub import hf_hub_download
    try:
        local = hf_hub_download(
            repo_id=HF_KNOW, filename='knowledge.json',
            repo_type='dataset', token=HF_TOKEN,
        )
    except Exception as e:
        print(f'  knowledge.json unavailable ({e}) — treating as empty.')
        return set()
    data = json.loads(Path(local).read_text(encoding='utf-8'))
    ids  = data.get('processed_contributions', [])
    return set(ids) if isinstance(ids, list) else set()


def _load_anchor_keys(api, repo_files: list[str]) -> set[tuple[str, str]]:
    """Set of (build_type, aspect_bucket) already promoted to data/anchors/."""
    from huggingface_hub import hf_hub_download
    keys: set[tuple[str, str]] = set()
    for f in repo_files:
        if not (f.startswith('data/anchors/') and f.endswith('.json')):
            continue
        try:
            local = hf_hub_download(
                repo_id=HF_DATASET, filename=f,
                repo_type='dataset', token=HF_TOKEN,
            )
            body = json.loads(Path(local).read_text(encoding='utf-8'))
        except Exception:
            continue
        bt  = (body.get('build_type') or '').strip()
        ab  = body.get('aspect_bucket')
        if bt and ab is not None:
            keys.add((bt, f'{float(ab):.2f}'))
    return keys


# ── drain planning ─────────────────────────────────────────────────────────────

def _plan_crops_drain(
    repo_files: list[str], promoted_shas: set[str], api,
) -> tuple[list[str], dict[str, list[dict]], dict[str, list[dict]]]:
    """Returns (png_paths_to_delete, ann_paths_to_rewrite, ann_paths_to_delete)."""
    from huggingface_hub import hf_hub_download

    # 1. staging crop PNGs whose sha is already promoted
    png_drops: list[str] = []
    for f in repo_files:
        if not (f.startswith('staging/') and f.endswith('.png')
                and '/crops/' in f):
            continue
        sha = Path(f).stem
        if sha in promoted_shas:
            png_drops.append(f)

    # 2. staging annotations.jsonl: trim promoted-sha lines
    ann_rewrites: dict[str, list[dict]] = {}
    ann_deletes:  list[str]             = []
    for f in repo_files:
        if not (f.startswith('staging/') and f.endswith('/annotations.jsonl')):
            continue
        try:
            local = hf_hub_download(
                repo_id=HF_DATASET, filename=f,
                repo_type='dataset', token=HF_TOKEN,
            )
        except Exception as e:
            print(f'  WARN: cannot fetch {f}: {e}')
            continue
        kept: list[dict] = []
        any_trimmed = False
        for line in Path(local).read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if (rec.get('crop_sha256') or '').strip() in promoted_shas:
                any_trimmed = True
                continue
            kept.append(rec)
        if not any_trimmed:
            continue
        if not kept:
            ann_deletes.append(f)
        else:
            ann_rewrites[f] = kept

    return png_drops, ann_rewrites, ann_deletes


def _plan_screens_drain(
    repo_files: list[str], promoted_shas: set[str],
) -> list[str]:
    """staging/<iid>/screen_types/<TYPE>/*.png whose sha is promoted."""
    drops: list[str] = []
    for f in repo_files:
        if not (f.startswith('staging/') and f.endswith('.png')
                and '/screen_types/' in f):
            continue
        sha = Path(f).stem
        if sha in promoted_shas:
            drops.append(f)
    return drops


def _plan_anchors_drain(
    repo_files: list[str], promoted_keys: set[tuple[str, str]], api,
) -> list[str]:
    """staging/<iid>/anchors_grid_*.json whose (build_type, aspect_bucket)
    already has data/anchors/<key>.json. OPT-IN — anchors aggregate votes
    per file, so dropping a staging file forfeits its slot contributions."""
    from huggingface_hub import hf_hub_download
    drops: list[str] = []
    for f in repo_files:
        name = Path(f).name
        if not (f.startswith('staging/') and name.startswith('anchors_grid_')
                and f.endswith('.json')):
            continue
        try:
            local = hf_hub_download(
                repo_id=HF_DATASET, filename=f,
                repo_type='dataset', token=HF_TOKEN,
            )
            body = json.loads(Path(local).read_text(encoding='utf-8'))
        except Exception:
            continue
        bt = (body.get('build_type') or '').strip()
        ab = body.get('aspect_bucket')
        if not (bt and ab is not None):
            continue
        key = (bt, f'{float(ab):.2f}')
        if key in promoted_keys:
            drops.append(f)
    return drops


def _plan_contributions_drain(
    repo_files: list[str], processed_ids: set[str],
) -> list[str]:
    """contributions/<date>/<id>.json (+ companion .png) for processed IDs."""
    drops: list[str] = []
    for f in repo_files:
        if not (f.startswith('contributions/') and f.endswith('.json')):
            continue
        cid = Path(f).stem
        if cid in processed_ids:
            drops.append(f)
            png = f[:-5] + '.png'
            if png in repo_files:
                drops.append(png)
    return drops


# ── commit ────────────────────────────────────────────────────────────────────

def _commit_deletes(
    api, repo_id: str,
    deletes: list[str],
    rewrites: dict[str, list[dict]],
    message: str,
) -> None:
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    import io
    ops: list = []
    for path in deletes:
        ops.append(CommitOperationDelete(path_in_repo=path))
    for path, recs in rewrites.items():
        buf = ('\n'.join(json.dumps(r, ensure_ascii=False) for r in recs)
               + '\n').encode('utf-8')
        ops.append(CommitOperationAdd(
            path_in_repo    = path,
            path_or_fileobj = io.BytesIO(buf),
        ))
    if not ops:
        print(f'  {repo_id}: nothing to do.')
        return
    print(f'  {repo_id}: committing {len(deletes)} deletes + '
          f'{len(rewrites)} rewrites…')
    # Chunked, additions before deletions — see hf_commit for why one
    # commit for everything is no longer safe at these volumes.
    commit_adds_then_deletes(
        api, repo_id, 'dataset', ops,
        message,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--apply', action='store_true',
                    help='Commit deletes (default: dry-run)')
    ap.add_argument('--include-anchors', action='store_true',
                    help='Also drain staging anchor files whose key already '
                         'has consensus. Lossy — only run after verifying '
                         'the active mergers have caught up.')
    args = ap.parse_args()

    if not HF_TOKEN:
        print('ERROR: HF_TOKEN is empty (set in .env or shell).', file=sys.stderr)
        return 2

    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)

    print(f'== Dataset repo: {HF_DATASET}')
    repo_files = _list_repo_files(api, HF_DATASET, 'dataset')
    print(f'   {len(repo_files)} paths listed.')

    print('-- Loading consensus sets from data/ …')
    promoted_crops   = _load_jsonl_field(api, HF_DATASET,
                                         'data/annotations.jsonl',
                                         'crop_sha256')
    promoted_screens = _load_jsonl_field(api, HF_DATASET,
                                         'data/screen_types/metadata.jsonl',
                                         'sha')
    print(f'   data/annotations.jsonl:           {len(promoted_crops)} shas')
    print(f'   data/screen_types/metadata.jsonl: {len(promoted_screens)} shas')

    promoted_anchors: set[tuple[str, str]] = set()
    if args.include_anchors:
        promoted_anchors = _load_anchor_keys(api, repo_files)
        print(f'   data/anchors/*.json:              {len(promoted_anchors)} keys')

    print('-- Planning dataset-repo drains …')
    png_drops, ann_rewrites, ann_deletes = _plan_crops_drain(
        repo_files, promoted_crops, api)
    screen_drops = _plan_screens_drain(repo_files, promoted_screens)
    anchor_drops: list[str] = []
    if args.include_anchors:
        anchor_drops = _plan_anchors_drain(repo_files, promoted_anchors, api)

    print(f'   crops:   {len(png_drops)} stage PNGs to delete, '
          f'{len(ann_rewrites)} ann files to trim, '
          f'{len(ann_deletes)} ann files to delete')
    print(f'   screens: {len(screen_drops)} stage PNGs to delete')
    if args.include_anchors:
        print(f'   anchors: {len(anchor_drops)} stage anchor files to delete')

    print(f'== Knowledge repo: {HF_KNOW}')
    know_files     = _list_repo_files(api, HF_KNOW, 'dataset')
    processed_ids  = _load_knowledge_processed(api)
    print(f'   {len(know_files)} paths listed, '
          f'{len(processed_ids)} processed contribution IDs.')
    contrib_drops  = _plan_contributions_drain(know_files, processed_ids)
    print(f'   contributions: {len(contrib_drops)} files to delete '
          f'({sum(1 for p in contrib_drops if p.endswith(".json"))} JSON + '
          f'{sum(1 for p in contrib_drops if p.endswith(".png"))} PNG)')

    total = (len(png_drops) + len(ann_deletes) + len(ann_rewrites)
             + len(screen_drops) + len(anchor_drops) + len(contrib_drops))
    print(f'-- Plan total: {total} operations across 2 repos')

    if not args.apply:
        print('DRY-RUN: pass --apply to commit.')
        return 0

    print('-- Committing dataset-repo drain …')
    _commit_deletes(
        api, HF_DATASET,
        deletes  = png_drops + ann_deletes + screen_drops + anchor_drops,
        rewrites = ann_rewrites,
        message  = (f'admin_drain_stale_staging: '
                    f'{len(png_drops)} crop PNGs, '
                    f'{len(ann_rewrites) + len(ann_deletes)} ann files, '
                    f'{len(screen_drops)} screen PNGs, '
                    f'{len(anchor_drops)} anchor files '
                    f'({datetime.now(UTC).strftime("%Y-%m-%d %H:%M")} UTC)'),
    )

    print('-- Committing knowledge-repo drain …')
    _commit_deletes(
        api, HF_KNOW,
        deletes  = contrib_drops,
        rewrites = {},
        message  = (f'admin_drain_stale_staging: '
                    f'{sum(1 for p in contrib_drops if p.endswith(".json"))} '
                    f'processed contributions '
                    f'({datetime.now(UTC).strftime("%Y-%m-%d %H:%M")} UTC)'),
    )

    print('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
