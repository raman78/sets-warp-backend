#!/usr/bin/env python3
"""Compare a local WARP CORE training store against the published dataset.

Every upload fault found so far has been silent in the same way: this machine
believed it had shared something the community dataset does not have, and no
routine ever put the two side by side. Each was found by hand, months late,
and only because a count looked wrong.

This is that comparison, run deliberately. It answers three questions the
client cannot answer alone, because each needs the published dataset:

    missing      confirmed here, absent there
    mislabelled  present there under a different label than this store holds
    withdrawn    present there, no longer here

`mislabelled` is the one that matters and the one nothing else reports. A
screenshot re-typed in WARP CORE kept its file, so its hash still matched, and
until 2026-09-05 the client skipped it as already sent — leaving the dataset
holding the label it was given the first time. Measured then: 26 of 27
screenshots typed `SPACE_BOFFS` here were published as `BOFFS`.

Read-only. It changes nothing and uploads nothing; fixing what it finds is
`admin_reject_crops.py` for labels, or letting the client re-sync for uploads.

Usage:
    python admin_reconcile_local.py --store ~/Shared/warp/training_data
    python admin_reconcile_local.py --store … --domain screens
    python admin_reconcile_local.py --store … --json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

from hf_clone import clone_hf_shallow

HF_ICONS_REPO_ID = os.environ.get('HF_ICONS_REPO_ID', 'sets-sto/sto-icon-dataset')
HF_TOKEN         = os.environ.get('HF_TOKEN', '')

# The client truncates its sha to 32 hex chars (`SyncWorker._file_sha256`), and
# the dataset is keyed on the same. Hashing the full digest here and comparing
# it against those keys finds nothing at all — which is a mistake worth naming,
# because the empty result reads as "everything agrees".
SHA_LEN = 32

# Not a label: the screen-type menu offers it so a user can undo a wrong pick,
# and the backend refuses it. Counting it would report a permanent backlog
# nobody can clear.
UNCLASSIFIED = 'UNKNOWN'


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()[:SHA_LEN]


def local_screens(store: Path) -> dict[str, str]:
    """`{sha: screen_type}` for every screenshot this store has classified."""
    out: dict[str, str] = {}
    root = store / 'screen_types'
    if not root.is_dir():
        return out
    for type_dir in sorted(root.iterdir()):
        if not type_dir.is_dir() or type_dir.name == UNCLASSIFIED:
            continue
        for png in type_dir.glob('*.png'):
            try:
                out[_sha(png)] = type_dir.name
            except Exception as e:
                print(f'  unreadable: {png.name} ({e})', file=sys.stderr)
    return out


def published_screens(snap: Path) -> dict[str, str]:
    """`{sha: screen_type}` as the merged dataset holds it."""
    out: dict[str, str] = {}
    meta = snap / 'data/screen_types/metadata.jsonl'
    if not meta.exists():
        return out
    for line in meta.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get('sha') and r.get('type'):
            out[r['sha']] = r['type']
    return out


def local_crops(store: Path) -> dict[str, str]:
    """`{sha: 'slot|name'}` for every confirmed crop this store holds.

    Read from `annotations.json` rather than from the crop filenames: the
    filename carries the label it had when the file was written, so a
    correction made later would be invisible exactly where it matters.
    """
    out: dict[str, str] = {}
    ann = store / 'annotations.json'
    crops = store / 'crops'
    if not ann.exists() or not crops.is_dir():
        return out
    try:
        data = json.loads(ann.read_text(encoding='utf-8'))
    except Exception as e:
        print(f'  annotations.json unreadable ({e})', file=sys.stderr)
        return out

    by_id: dict[str, str] = {}
    for rec in data.values():
        if not isinstance(rec, dict):
            continue
        for a in rec.get('annotations') or []:
            if isinstance(a, dict) and a.get('ann_id') and a.get('name'):
                by_id[str(a['ann_id'])] = f"{a.get('slot', '')}|{a['name']}"

    for png in crops.rglob('*.png'):
        # `<slot>__<name>__<ann_id>.png` — the id is the last field.
        ann_id = png.stem.rsplit('__', 1)[-1]
        label = by_id.get(ann_id)
        if label:
            try:
                out[_sha(png)] = label
            except Exception:
                pass
    return out


def published_crops(snap: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    p = snap / 'data/annotations.jsonl'
    if not p.exists():
        return out
    for line in p.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get('crop_sha256') and r.get('name'):
            out[r['crop_sha256'][:SHA_LEN]] = f"{r.get('slot', '')}|{r['name']}"
    return out


def compare(local: dict[str, str], remote: dict[str, str]) -> dict[str, list]:
    """Split the two views into the three states worth acting on."""
    missing     = sorted(sha for sha in local if sha not in remote)
    withdrawn   = sorted(sha for sha in remote if sha not in local)
    mislabelled = sorted((sha, local[sha], remote[sha])
                         for sha in local if sha in remote
                         and local[sha] != remote[sha])
    return {'missing': missing, 'mislabelled': mislabelled,
            'withdrawn': withdrawn}


def _report(domain: str, local: dict, remote: dict, verdict: dict) -> None:
    print(f'\n=== {domain} ===')
    print(f'  here {len(local)}   published {len(remote)}')
    print(f'  missing      {len(verdict["missing"]):5}  confirmed here, absent there')
    print(f'  mislabelled  {len(verdict["mislabelled"]):5}  published under a different label')
    print(f'  withdrawn    {len(verdict["withdrawn"]):5}  published, no longer here')

    if verdict['mislabelled']:
        pairs = collections.Counter((l, r) for _, l, r in verdict['mislabelled'])
        print('\n  published under a different label:')
        for (l, r), n in pairs.most_common(15):
            print(f'    {n:5}  here {l!r}\n           there {r!r}')

    if verdict['missing']:
        by_label = collections.Counter(local[s] for s in verdict['missing'])
        print('\n  confirmed here but absent there:')
        for lab, n in by_label.most_common(15):
            print(f'    {n:5}  {lab!r}')


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--store', required=True, type=Path,
                    help='WARP CORE training store (the folder holding '
                         'annotations.json and screen_types/)')
    ap.add_argument('--domain', choices=('screens', 'crops', 'both'),
                    default='both')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    if not args.store.is_dir():
        print(f'ERROR: no such store: {args.store}', file=sys.stderr)
        return 2
    if not HF_TOKEN:
        print('ERROR: HF_TOKEN is not set.', file=sys.stderr)
        return 2

    snap = clone_hf_shallow(HF_ICONS_REPO_ID, HF_TOKEN, repo_type='dataset')

    result: dict[str, dict] = {}
    if args.domain in ('screens', 'both'):
        l, r = local_screens(args.store), published_screens(snap)
        result['screens'] = {'local': len(l), 'published': len(r),
                             **compare(l, r)}
        if not args.json:
            _report('screen types', l, r, result['screens'])
    if args.domain in ('crops', 'both'):
        l, r = local_crops(args.store), published_crops(snap)
        result['crops'] = {'local': len(l), 'published': len(r), **compare(l, r)}
        if not args.json:
            _report('crops', l, r, result['crops'])

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    disagreements = sum(len(d['missing']) + len(d['mislabelled'])
                        for d in result.values())
    print(f'\n{disagreements} item(s) this machine and the dataset disagree on.'
          if disagreements else '\nThis machine and the dataset agree.')
    # Exit 1 on any disagreement so a scripted run can act on it; `withdrawn`
    # is excluded because a maintainer rejection is a legitimate reason for
    # the dataset to hold something this store no longer does.
    return 1 if disagreements else 0


if __name__ == '__main__':
    sys.exit(main())
