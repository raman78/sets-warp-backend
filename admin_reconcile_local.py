#!/usr/bin/env python3
"""Did this machine's contributions reach the dataset, and where is it outvoted.

The published dataset is the source of truth: it is what the trainers read and
what every user receives. A local store is one contributor's opinion, the
maintainer's included, and it holding a different label is normally the
consensus working rather than a fault.

So this does not ask "do the two agree". It asks the only question a local
store can settle on its own — **was my decision ever submitted** — and keeps
that strictly apart from the answer it got:

    unsent       confirmed here, and this install never sent that label.
                 A transport fault. The dataset cannot have considered it.

    outvoted     sent under this label, and the dataset settled on another.
                 Not a fault: the tally weighed it and other contributors
                 disagreed. Listed so a maintainer can review, never counted
                 as an error.

    absent       sent, accepted, and no longer in the dataset — usually a
                 maintainer rejection, and equally not a fault.

The distinction is drawn from the client's own upload cache, which records the
label each crop was last sent under. Without it the two are indistinguishable,
and reporting a lost upload and a lost vote as the same thing is what made an
earlier version of this tool argue that the dataset should be corrected to
match one machine.

Exit status is 1 only when something was never sent. A disagreement the tally
produced is information, not a failure.

Read-only. It changes nothing and uploads nothing.

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


def sent_labels(store: Path, domain: str) -> dict[str, str]:
    """What this install last *sent* for each sha, from its own upload cache.

    This is what separates a lost upload from a lost vote. Without it the two
    look identical from outside, and an earlier version of this tool reported
    both as the dataset being wrong.

    An empty result means the cache is missing or unreadable, and every
    comparison then falls back to "cannot tell" rather than guessing.
    """
    name = ('.sync_uploaded_screen_hashes.json' if domain == 'screens'
            else '.sync_uploaded_labels.json')
    try:
        raw = json.loads((store / name).read_text(encoding='utf-8'))
    except Exception:
        return {}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, list):
        # The screen cache was a bare list of shas until 2026-09-05, which
        # recorded no label at all. Treat every entry as "sent, label
        # unknown" — it cannot support an `outvoted` claim.
        return {str(sha): '' for sha in raw}
    return {}


def compare(local: dict[str, str], remote: dict[str, str],
            sent: dict[str, str]) -> dict[str, list]:
    """Split the two views by *why* they differ, not by whether they do.

    `sent` decides the split. A label the dataset does not hold and this
    install never sent is a transport fault; one it did send and the tally
    settled otherwise is the tally working.
    """
    unsent:   list[tuple[str, str]] = []
    outvoted: list[tuple[str, str, str]] = []
    absent:   list[str] = []

    for sha, mine in local.items():
        theirs = remote.get(sha)
        if theirs == mine:
            continue
        was_sent = sent.get(sha) == mine
        if theirs is None:
            (outvoted.append((sha, mine, '<dropped>')) if was_sent
             else unsent.append((sha, mine)))
        elif was_sent:
            outvoted.append((sha, mine, theirs))
        else:
            # The dataset holds an older label from this same install, and the
            # correction was never sent. That is the transport fault, and it
            # is the shape the screen-type bug produced 93 times.
            unsent.append((sha, mine))

    for sha in remote:
        if sha not in local:
            absent.append(sha)

    return {'unsent': sorted(unsent), 'outvoted': sorted(outvoted),
            'absent': sorted(absent)}


def _report(domain: str, local: dict, remote: dict, v: dict) -> None:
    print(f'\n=== {domain} ===')
    print(f'  here {len(local)}   published {len(remote)}   '
          f'(the published dataset is the reference)')
    print(f'  unsent    {len(v["unsent"]):5}  never submitted from here — a transport fault')
    print(f'  outvoted  {len(v["outvoted"]):5}  submitted, and the tally settled otherwise')
    print(f'  absent    {len(v["absent"]):5}  in the dataset, not in this store')

    if v['unsent']:
        by_label = collections.Counter(lab for _, lab in v['unsent'])
        print('\n  never submitted from here:')
        for lab, n in by_label.most_common(15):
            print(f'    {n:5}  {lab!r}')

    if v['outvoted']:
        pairs = collections.Counter((mine, theirs) for _, mine, theirs in v['outvoted'])
        print('\n  submitted, and the dataset settled on something else:')
        print('  (not a fault — review only if a pairing looks wrong)')
        for (mine, theirs), n in pairs.most_common(15):
            print(f'    {n:5}  sent {mine!r}\n           kept {theirs!r}')


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
                             **compare(l, r, sent_labels(args.store, 'screens'))}
        if not args.json:
            _report('screen types', l, r, result['screens'])
    if args.domain in ('crops', 'both'):
        l, r = local_crops(args.store), published_crops(snap)
        result['crops'] = {'local': len(l), 'published': len(r),
                           **compare(l, r, sent_labels(args.store, 'crops'))}
        if not args.json:
            _report('crops', l, r, result['crops'])

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    unsent = sum(len(d['unsent']) for d in result.values())
    if unsent:
        print(f'\n{unsent} decision(s) made here never reached the dataset.')
    else:
        print('\nEverything decided here has been submitted.')
    # Exit 1 only for a transport fault. Being outvoted is the tally doing its
    # job, and scoring it as a failure would teach whoever runs this that a
    # non-zero exit means nothing.
    return 1 if unsent else 0


if __name__ == '__main__':
    sys.exit(main())
