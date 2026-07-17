#!/usr/bin/env python3
"""
admin_reject_crops.py — review + purge mislabeled virtual crops from data/
==========================================================================
Companion to democratic_merge_crops.py. Operates on the published mirror of
the crop dataset:

    sets-sto/sto-icon-dataset    data/annotations.jsonl + data/crops/<sha>.png

Why this tool exists
--------------------
`__empty__` / `__inactive__` are legitimate ML labels (the ArcFace embedder
needs them as gallery classes), so `democratic_merge_crops.py` deliberately
lets them through. But WARP CORE auto-accept sometimes tags a *colourful,
real* icon as an empty/inactive slot on a low-confidence detection the user
never corrected. Those crops are poison: they teach the embedder that a real
icon is "empty", and the client-side visual guard
(`icon_matcher._virtual_crop_looks_real`) logs them every seed as
`CommunitySeed: POISON skip`.

This tool surfaces exactly those crops (virtual label + colourful pixels),
lets the maintainer review each one, and applies a three-way decision:

    KEEP           the label is correct — a real dim slot that just happens
                   to trip the heuristic; recorded so it is never re-surfaced.
    REJECT         drop it entirely — bad training data, unidentifiable.
    RELABEL <name> same crop bytes (same sha), fix the `name` to the real
                   item so the poison becomes useful training data.

A per-crop review ledger (`data/reviewed_virtual.jsonl`) records every
decision so re-scans (and the cron audit / GUI dashboard) only ever show
NEW, unreviewed poison — you never re-litigate a KEEP.

Visual heuristic
----------------
The bright/rich ratios below MUST stay in sync with
`sto-warp:warp/recognition/icon_matcher.py:_virtual_crop_looks_real` and
`warp.tools.scrub_training_data` — same numbers so this tool flags exactly
what the client rejects.

Workflow
--------
    # 1. scan (dry-run) — writes a montage PNG + a decisions TSV
    python admin_reject_crops.py

    # 2. open the montage, edit the TSV: change REJECT -> KEEP or
    #    'RELABEL <canonical name>' for the crops you recognise

    # 3. apply — one atomic HF commit (data/ purge/relabel + staging drain
    #    + ledger append)
    python admin_reject_crops.py --apply

Environment (.env, same as the mergers):
    HF_TOKEN     — HF write token (required for --apply; read-only scan too)
    HF_DATASET   — default: sets-sto/sto-icon-dataset
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# Reuse HF helpers + .env loader from admin_train.py (single source of truth).
sys.path.insert(0, str(Path(__file__).parent))
from admin_train import (  # noqa: E402
    HF_TOKEN, HF_DATASET,
    _require_hf, _create_commit_with_retry,
)

UTC = timezone.utc

REPO  = HF_DATASET
RTYPE = 'dataset'

DATA_ANN = 'data/annotations.jsonl'
DATA_CRP = 'data/crops'
LEDGER   = 'data/reviewed_virtual.jsonl'

VIRTUAL_LABELS = frozenset({'__empty__', '__inactive__'})

# KEEP IN SYNC with sto-warp icon_matcher._virtual_crop_looks_real /
# warp.tools.scrub_training_data. A virtual-labeled crop with BOTH more than
# these fractions of bright (V>150) and colour-rich (S>100 & V>100) pixels is
# a real icon mislabeled empty/inactive.
VIRTUAL_SEED_BRIGHT_RATIO = 0.15
VIRTUAL_SEED_RICH_RATIO   = 0.15

DEFAULT_MONTAGE   = 'virtual_reject_montage.png'
DEFAULT_DECISIONS = 'virtual_reject_decisions.tsv'
# Local sibling checkout of sto-warp (maintainer layout: repos side by side).
# Used as an import fallback when sto-warp isn't pip-installed into this venv.
_STO_WARP_SIBLING = Path(__file__).resolve().parent.parent / 'sto-warp'
_MONTAGE_CELL     = 96   # px per crop in the review montage
_MONTAGE_COLS     = 6


# ── Visual heuristic ───────────────────────────────────────────────────────────

def _bright_rich(bgr: np.ndarray) -> tuple[float, float]:
    """Return (bright, rich) pixel fractions — mirror of the client heuristic."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    bright = float((v > 150).mean())
    rich   = float(((s > 100) & (v > 100)).mean())
    return bright, rich


def _looks_real(bgr: np.ndarray) -> bool:
    bright, rich = _bright_rich(bgr)
    return bright > VIRTUAL_SEED_BRIGHT_RATIO and rich > VIRTUAL_SEED_RICH_RATIO


# ── Dataset readers ─────────────────────────────────────────────────────────────

def _load_jsonl_by_sha(path: Path) -> dict[str, dict]:
    """Load a <sha>-keyed annotations JSONL (last write wins)."""
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            sha = d.get('crop_sha256')
            if sha:
                out[sha] = d
    return out


def _load_ledger(snap_dir: Path) -> dict[str, dict]:
    """Load data/reviewed_virtual.jsonl → {sha: {name, decision, reviewed_at}}."""
    return _load_jsonl_by_sha(snap_dir / LEDGER)


def load_canonical_names() -> set[str]:
    """Valid RELABEL targets, taken straight from sto-warp's own cargo loader
    (`warp.data.cargo.canonical_names`) — the single source of truth. We do
    NOT re-parse the cargo JSON here: sto-warp already fetches, caches, and
    sanitizes it, so importing the source keeps this tool in lock-step with
    what the client actually recognises. Prefers a pip-installed sto-warp;
    falls back to the sibling repo checkout for the local maintainer layout.
    Empty set means sto-warp is unavailable — callers must then refuse to
    relabel rather than guess."""
    try:
        from warp.data.cargo import canonical_names
    except Exception:
        if (_STO_WARP_SIBLING / 'warp' / 'data' / 'cargo.py').exists():
            sys.path.insert(0, str(_STO_WARP_SIBLING))
        try:
            from warp.data.cargo import canonical_names
        except Exception as e:
            print(f'WARNING: sto-warp cargo unavailable ({e}). Install the admin '
                  f'extra (pip install -e ".[admin]") or place the sto-warp repo '
                  f'beside this one.', file=sys.stderr)
            return set()
    try:
        return canonical_names()
    except Exception as e:
        print(f'WARNING: cargo.canonical_names() failed: {e}', file=sys.stderr)
        return set()


# ── Scan ────────────────────────────────────────────────────────────────────────

def local_mirror_crops_dir() -> Path | None:
    """Path to sto-warp's already-downloaded community crop mirror, if present.

    sto-warp keeps the full dataset mirrored with REAL pixels under
    `~/.cache/warp/community_crops/data/crops/` (this is what the client seeds
    from and what `diag_view_community_poison.py` reads). On a maintainer
    machine that runs sto-warp, reading pixels from here avoids re-downloading
    hundreds of crops from HF — the shallow clone only carries LFS stubs."""
    try:
        from warp.knowledge.community_crops import community_crops_dir
    except Exception:
        if (_STO_WARP_SIBLING / 'warp' / 'knowledge' / 'community_crops.py').exists():
            sys.path.insert(0, str(_STO_WARP_SIBLING))
        try:
            from warp.knowledge.community_crops import community_crops_dir
        except Exception:
            return None
    try:
        d = community_crops_dir()
        return d if d.exists() else None
    except Exception:
        return None


def _fetch_crop(sha: str, token: str,
                local_dir: Path | None = None) -> np.ndarray | None:
    """Decode data/crops/<sha>.png. Prefers sto-warp's local mirror (real
    pixels, no network); falls back to hf_hub_download only for shas the
    mirror lacks. The shallow clone itself holds Git-LFS stubs, not images."""
    if local_dir is not None:
        p = local_dir / f'{sha}.png'
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return img
    from huggingface_hub import hf_hub_download
    try:
        local = hf_hub_download(
            repo_id=REPO, repo_type=RTYPE, token=token,
            filename=f'{DATA_CRP}/{sha}.png',
        )
    except Exception:
        return None
    return cv2.imread(local)


def scan(snap_dir: Path,
         token: str,
         bright_ratio: float,
         rich_ratio: float,
         show_reviewed: bool) -> list[dict]:
    """Return colourful virtual-label crops in data/ that need review.

    Reads annotations from the shallow clone (plain text) to find every
    __empty__/__inactive__ sha, then fetches only those crop PNGs via
    hf_hub_download (the clone holds LFS stubs, not pixels) and applies the
    bright/rich heuristic. Skips shas already decided KEEP in the ledger;
    `show_reviewed=True` includes them (annotated with their prior decision).
    """
    global VIRTUAL_SEED_BRIGHT_RATIO, VIRTUAL_SEED_RICH_RATIO
    VIRTUAL_SEED_BRIGHT_RATIO = bright_ratio
    VIRTUAL_SEED_RICH_RATIO   = rich_ratio

    data = _load_jsonl_by_sha(snap_dir / DATA_ANN)
    ledger = _load_ledger(snap_dir)

    virtual = [(sha, rec) for sha, rec in data.items()
               if (rec.get('name') or '').strip() in VIRTUAL_LABELS]
    local_dir = local_mirror_crops_dir()
    src = (f'sto-warp local mirror ({local_dir})' if local_dir
           else 'HF download (no local mirror found)')
    print(f'Virtual-label crops in data/: {len(virtual)} — reading pixels from '
          f'{src}…')

    out: list[dict] = []
    missing = 0
    for i, (sha, rec) in enumerate(virtual, 1):
        if i % 200 == 0:
            print(f'  … {i}/{len(virtual)}')
        name = (rec.get('name') or '').strip()

        # Only a KEEP is final: a rejected/relabeled sha that reappears as a
        # colourful virtual is a resurrection (a re-upload the denylist has
        # not yet caught) and must be re-surfaced, not silently skipped.
        prior = ledger.get(sha)
        already_decided = bool(prior and prior.get('decision') == 'KEEP'
                               and (prior.get('name') or '') == name)
        if already_decided and not show_reviewed:
            continue

        img = _fetch_crop(sha, token, local_dir=local_dir)
        if img is None:
            missing += 1
            continue
        if not _looks_real(img):
            continue

        bright, rich = _bright_rich(img)
        out.append({
            'sha':    sha,
            'name':   name,
            'slot':   rec.get('slot', ''),
            'bright': bright,
            'rich':   rich,
            'img':    img,
            'prior':  (prior or {}).get('decision', '') if prior else '',
        })
    if missing:
        print(f'  WARN: {missing} virtual crop(s) could not be fetched/decoded.')
    out.sort(key=lambda e: e['rich'], reverse=True)
    return out


# ── Montage + decisions TSV ─────────────────────────────────────────────────────

def write_montage(candidates: list[dict], out_path: Path) -> None:
    """Grid PNG (index + sha prefix per cell) so the maintainer can eyeball
    every flagged crop against the row order of the decisions TSV."""
    cell = _MONTAGE_CELL
    cols = _MONTAGE_COLS
    rows = (len(candidates) + cols - 1) // cols
    canvas = np.full((rows * (cell + 26), cols * (cell + 8), 3), 30, np.uint8)
    for i, e in enumerate(candidates):
        r, c = divmod(i, cols)
        y = r * (cell + 26) + 22
        x = c * (cell + 8) + 4
        thumb = cv2.resize(e['img'], (cell, cell), interpolation=cv2.INTER_NEAREST)
        canvas[y:y + cell, x:x + cell] = thumb
        tag = f'{i + 1} {e["sha"][:6]}'
        cv2.putText(canvas, tag, (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def write_decisions_tsv(candidates: list[dict], out_path: Path) -> None:
    """Pre-fill a REJECT decision per crop. The maintainer edits the first
    column to KEEP or 'RELABEL <name>' before --apply."""
    lines = ['# decision\tidx\tsha\tlabel\tslot\tbright\trich',
             '# decision ∈ {REJECT, KEEP, RELABEL <canonical name>}']
    for i, e in enumerate(candidates, 1):
        lines.append(
            f'REJECT\t{i}\t{e["sha"]}\t{e["name"]}\t{e["slot"]}\t'
            f'{e["bright"]:.3f}\t{e["rich"]:.3f}'
        )
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def read_decisions_tsv(path: Path) -> list[dict]:
    """Parse the edited TSV → [{sha, decision, relabel_name}]."""
    decisions: list[dict] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.rstrip('\n')
        if not line or line.lstrip().startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) < 3:
            continue
        raw = parts[0].strip()
        sha = parts[2].strip()
        if not sha:
            continue
        tok = raw.split(None, 1)
        verb = tok[0].upper()
        relabel_name = tok[1].strip() if verb == 'RELABEL' and len(tok) > 1 else ''
        if verb not in ('REJECT', 'KEEP', 'RELABEL'):
            print(f'  WARN: unknown decision {raw!r} for {sha[:10]} — skipping row')
            continue
        if verb == 'RELABEL' and not relabel_name:
            print(f'  WARN: RELABEL without a name for {sha[:10]} — skipping row')
            continue
        decisions.append({'sha': sha, 'decision': verb, 'relabel_name': relabel_name})
    return decisions


# ── Apply ────────────────────────────────────────────────────────────────────────

def apply(snap_dir: Path, decisions: list[dict], api, repo_files: set[str],
          canonical: set[str]) -> bool:
    """One atomic HF commit: purge/relabel data/, drain staging copies of
    rejected shas, and append every decision to the review ledger.

    RELABEL targets are validated against `canonical` (the sto-warp cargo
    list) — an unknown name aborts the whole run before any commit, so a
    typo can never enter the dataset."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    data = _load_jsonl_by_sha(snap_dir / DATA_ANN)
    by_sha = {d['sha']: d for d in decisions}

    reject = {s for s, d in by_sha.items() if d['decision'] == 'REJECT'}
    relabel = {s: d['relabel_name'] for s, d in by_sha.items()
               if d['decision'] == 'RELABEL'}
    keep = {s for s, d in by_sha.items() if d['decision'] == 'KEEP'}

    # Guard: RELABEL names must exist in the cargo list. Refuse the whole
    # commit otherwise — never write a hand-typed / mistyped label.
    if relabel:
        if not canonical:
            print('ERROR: cargo list is empty — cannot validate RELABEL names. '
                  'Point --cargo-dir at a populated sto-warp cache.',
                  file=sys.stderr)
            return False
        bad = {s: n for s, n in relabel.items() if n not in canonical}
        if bad:
            print('ERROR: RELABEL name(s) not in cargo — aborting (no commit):',
                  file=sys.stderr)
            for s, n in bad.items():
                print(f'  {s[:12]}  {n!r}', file=sys.stderr)
            return False

    unknown = [s for s in by_sha if s not in data]
    if unknown:
        print(f'  WARN: {len(unknown)} decided sha not in data/annotations.jsonl '
              f'(already gone?) — ignored')

    # 1. Rewrite data/annotations.jsonl: drop REJECT, apply RELABEL name.
    new_data: dict[str, dict] = {}
    for sha, rec in data.items():
        if sha in reject:
            continue
        if sha in relabel:
            rec = dict(rec)
            rec['name'] = relabel[sha]
            rec['relabeled_at'] = _now_iso()
        new_data[sha] = rec
    ann_payload = ('\n'.join(json.dumps(new_data[s], ensure_ascii=False)
                             for s in sorted(new_data)) + '\n').encode('utf-8')

    ops: list = [CommitOperationAdd(path_in_repo=DATA_ANN,
                                    path_or_fileobj=io.BytesIO(ann_payload))]

    # 2. Delete rejected crop PNGs from data/crops/.
    deleted_crops = 0
    for sha in reject:
        dst = f'{DATA_CRP}/{sha}.png'
        if dst in repo_files:
            ops.append(CommitOperationDelete(path_in_repo=dst))
            deleted_crops += 1

    # 3. Drain any lingering staging copies of rejected shas so the next
    #    democratic_merge run cannot re-promote them.
    drained = 0
    for sha in reject:
        for f in repo_files:
            if f.startswith('staging/') and f.endswith(f'/crops/{sha}.png'):
                ops.append(CommitOperationDelete(path_in_repo=f))
                drained += 1

    # 4. Append every decision to the review ledger (append-only, last wins).
    ledger = _load_ledger(snap_dir)
    for sha, d in by_sha.items():
        ledger[sha] = {
            'crop_sha256': sha,
            'name':        (relabel.get(sha) if sha in relabel
                            else (data.get(sha, {}).get('name') or '')),
            'decision':    d['decision'],
            'reviewed_at': _now_iso(),
        }
    ledger_payload = ('\n'.join(json.dumps(ledger[s], ensure_ascii=False)
                                for s in sorted(ledger)) + '\n').encode('utf-8')
    ops.append(CommitOperationAdd(path_in_repo=LEDGER,
                                  path_or_fileobj=io.BytesIO(ledger_payload)))

    msg = (f'admin_reject_crops: reject {len(reject)}, relabel {len(relabel)}, '
           f'keep {len(keep)} virtual crops '
           f'(-{deleted_crops} crops, -{drained} staging) '
           f'@ {datetime.now(UTC).strftime("%Y-%m-%d %H:%M")} UTC')
    print(f'Committing: {msg}')
    return _create_commit_with_retry(api, REPO, RTYPE, ops, msg)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec='seconds').replace('+00:00', 'Z')


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description='Review + purge mislabeled virtual crops from data/.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--apply', action='store_true',
                    help='Read the decisions TSV and commit to HF '
                         '(default: dry-run scan → montage + TSV).')
    ap.add_argument('--decisions', default=DEFAULT_DECISIONS,
                    help=f'Decisions TSV path (default: {DEFAULT_DECISIONS}).')
    ap.add_argument('--montage', default=DEFAULT_MONTAGE,
                    help=f'Review montage PNG path (default: {DEFAULT_MONTAGE}).')
    ap.add_argument('--bright-ratio', type=float, default=VIRTUAL_SEED_BRIGHT_RATIO,
                    help='Bright-pixel fraction gate (keep in sync with client).')
    ap.add_argument('--rich-ratio', type=float, default=VIRTUAL_SEED_RICH_RATIO,
                    help='Colour-rich fraction gate (keep in sync with client).')
    ap.add_argument('--show-reviewed', action='store_true',
                    help='Include crops already decided in the ledger.')
    args = ap.parse_args()

    _require_hf()

    print('=' * 64)
    print(f'WARP virtual-crop review — {REPO}')
    print(f'Mode: {"APPLY" if args.apply else "SCAN"}  ·  '
          f'gate bright>{args.bright_ratio} rich>{args.rich_ratio}')
    print('=' * 64)

    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)

    print('Cloning repo (shallow)…')
    from hf_clone import clone_hf_shallow
    snap_dir = clone_hf_shallow(REPO, HF_TOKEN, repo_type=RTYPE)
    repo_files = set(subprocess.check_output(
        ['git', 'ls-files'], cwd=str(snap_dir), text=True).splitlines())

    if args.apply:
        dpath = Path(args.decisions)
        if not dpath.exists():
            print(f'ERROR: decisions file not found: {dpath}\n'
                  f'Run a scan first (no --apply) to generate it.', file=sys.stderr)
            return 2
        decisions = read_decisions_tsv(dpath)
        if not decisions:
            print('No valid decisions parsed — nothing to do.')
            return 0
        n = {v: sum(1 for d in decisions if d['decision'] == v)
             for v in ('REJECT', 'KEEP', 'RELABEL')}
        print(f'Parsed {len(decisions)} decisions: '
              f'{n["REJECT"]} reject · {n["KEEP"]} keep · {n["RELABEL"]} relabel')
        canonical = load_canonical_names()
        print(f'Cargo names for RELABEL validation: {len(canonical)} '
              f'(from sto-warp warp.data.cargo)')
        ok = apply(snap_dir, decisions, api, repo_files, canonical)
        print('OK — committed.' if ok else 'FAILED — see errors above.')
        return 0 if ok else 1

    # Scan (dry-run)
    candidates = scan(snap_dir, HF_TOKEN, args.bright_ratio, args.rich_ratio,
                      show_reviewed=args.show_reviewed)
    print(f'\nFlagged {len(candidates)} colourful virtual crop(s) needing review:')
    for i, e in enumerate(candidates, 1):
        prior = f'  (ledger: {e["prior"]})' if e['prior'] else ''
        print(f'  [{i:>2}] {e["sha"][:10]}  {e["name"]:<12} '
              f'slot={e["slot"]!r:<18} bright={e["bright"]:.1%} '
              f'rich={e["rich"]:.1%}{prior}')

    if not candidates:
        print('Nothing to review — data/ is clean of colourful virtual crops.')
        return 0

    write_montage(candidates, Path(args.montage))
    write_decisions_tsv(candidates, Path(args.decisions))
    print(f'\nMontage  → {args.montage}   (open it to eyeball each crop)')
    print(f'Decisions → {args.decisions}   '
          f'(edit REJECT→KEEP or "RELABEL <name>", then run --apply)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
