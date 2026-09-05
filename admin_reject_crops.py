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
lets the maintainer review each one, and applies a four-way decision:

    KEEP           the label is correct — a real dim slot that just happens
                   to trip the heuristic; recorded so it is never re-surfaced.
    REJECT         drop it entirely — bad training data, unidentifiable.
                   Note this is **permanent**: the sha is barred and a later
                   re-upload of the same picture can never be promoted, so it
                   is the wrong tool for a crop that is merely mis-filed.
    RELABEL <name> same crop bytes (same sha), fix the `name` to the real
                   item so the poison becomes useful training data.
    SLOT <slot>    same crop, same name, wrong *slot*. Nothing is deleted and
                   nothing is barred.

`SLOT` exists because of a defect that is now fixed upstream but left records
behind. When no tier badge was found on screen, the client gave the `Ship
Tier` row the same bounding box as the `Ship Type` row; identical pixels give
an identical hash, so both rows landed in one ballot and a record could end up
with a class line's picture, the class's name, and `slot: Ship Tier`.
Rejecting such a record would throw away a perfectly good picture of a ship's
class line — often the only copy — and bar it for good. Only one field is
wrong, so only one field is changed.

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

# HF refuses a push that would leave more than 10 000 files in one directory,
# and `data/crops/` filled up: the promotion froze on 2026-07-16 with the
# folder at ~9 985 and every run since died on a 400 the server would not
# explain until the message was read in full. New crops go under a two-hex
# shard of their own sha. The flat files predate it and are migrated in their
# own pass, so both layouts have to be readable meanwhile.

def crop_path(sha: str) -> str:
    """Where a crop is written today."""
    return f'{DATA_CRP}/{sha[:2]}/{sha}.png'


def crop_paths(sha: str) -> tuple[str, str]:
    """Both places a crop may live — sharded first, then the legacy flat one."""
    return crop_path(sha), f'{DATA_CRP}/{sha}.png'

LEDGER   = 'data/reviewed_virtual.jsonl'

VIRTUAL_LABELS = frozenset({'__empty__', '__inactive__'})
# Text crops (the ship name / class / tier bands) are wide low-contrast
# strips, not slot cells — the blank-cell judgement does not apply to
# them. Mirrors `_TEXT_CROP_PREFIXES` in main.py.
_TEXT_CROP_SLOT_PREFIXES = ('ship_type', 'ship_tier', 'Ship Type',
                            'Ship Tier', 'Ship Name')

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


def _looks_blank(bgr: np.ndarray) -> bool:
    """The mirror case: a crop under a real item's name that is an empty or
    inactive cell.

    More damaging than a colourful virtual, because it teaches the gallery
    that the item *is* what nothing looks like, and the recogniser then
    answers with that item on every blank cell. Measured 2026-09-03 on the
    published mirror: 25 of 9227 real-named crops are blank, and 20 of them
    carry one name — `Charged Particle Burst`, 20 of the 29 crops that class
    has. An inactive BOFF cell sits at cosine 0.92 from those 20 and 0.45
    from the 9 genuine ones.

    Delegated to sto-warp's `_real_crop_looks_blank` for the same reason
    `load_canonical_names` delegates: one definition, so this tool flags
    exactly what the client refuses to seed. Without sto-warp available the
    check is off rather than approximated.
    """
    fn = _load_blank_check()
    return bool(fn(bgr)) if fn else False


def _load_blank_check():
    try:
        from warp.recognition.icon_matcher import _real_crop_looks_blank
        return _real_crop_looks_blank
    except Exception:
        if (_STO_WARP_SIBLING / 'warp' / 'recognition' / 'icon_matcher.py').exists():
            sys.path.insert(0, str(_STO_WARP_SIBLING))
        try:
            from warp.recognition.icon_matcher import _real_crop_looks_blank
            return _real_crop_looks_blank
        except Exception as e:
            print(f'WARNING: sto-warp unavailable ({e}) — the blank-cell '
                  f'direction is skipped.', file=sys.stderr)
            return None


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


_SHIP_TIER_VALUES = frozenset({
    'T1', 'T2', 'T3', 'T4', 'T5', 'T6',
    'T5-U', 'T5-X', 'T6-X', 'T6-X2',
})


def load_vocabularies() -> dict[str, set[str]]:
    """The name sets a crop's label can legitimately belong to, by kind.

    Three, because a crop's slot decides which one applies: icon slots hold
    item names, a `Ship Type` band holds a ship name, a `Ship Tier` band holds
    a tier token. Sourced from sto-warp for the same reason
    `load_canonical_names` is — one definition of what the client recognises.

    A missing set means "cannot check", and `_name_resolves_nowhere` then
    declines to flag rather than guessing.
    """
    vocab: dict[str, set[str]] = {'items': load_canonical_names(),
                                  'ships': set(),
                                  'tiers': set(_SHIP_TIER_VALUES)}
    try:
        from warp.data.cargo import ships
        # Both names each ship has, because a `Ship Type` crop is OCR of what
        # the *game* prints. `ships()` is keyed on `Page`, the wiki article
        # title, and the two differ for 84 of the 797 ships — the Galaxy
        # Retrofit's article is `Galaxy Exploration Cruiser Retrofit` while
        # the game shows `Exploration Cruiser Retrofit`. Keying the check on
        # the article title alone flags every one of those as unresolvable,
        # and worse, makes the RELABEL guard refuse a correction to the name
        # the ship actually displays.
        rows = ships()
        vocab['ships'] = set(rows.keys()) | {
            (r.get('name') or '').strip()
            for r in rows.values() if isinstance(r, dict) and r.get('name')
        }
    except Exception as e:
        print(f'WARNING: ship list unavailable ({e}) — ship-name labels '
              f'will not be checked.', file=sys.stderr)
    return vocab


def _name_resolves_nowhere(name: str, slot: str, vocab: dict[str, set[str]]) -> bool:
    """True if `name` is in none of the vocabularies its slot allows.

    Ship Type is checked against the ship roster *and* the item names: the
    slot is assigned by the detector and a mislabelled row is exactly what
    this hunts, so a real item sitting in a text slot should not be reported
    as an unresolvable name.
    """
    # The virtual classes are labels in their own right — the embedder needs
    # them as gallery classes, and this tool defines them a few lines up. No
    # cargo table carries them, so an item-name check refuses them and the
    # commonest correction there is, a blank cell filed under an item's name,
    # could not be applied at all.
    if name in VIRTUAL_LABELS:
        return False

    slot = (slot or '').strip()
    if slot.startswith(('ship_tier', 'Ship Tier')):
        pool = vocab['tiers'] | vocab['items']
    elif slot.startswith(('ship_type', 'Ship Type', 'Ship Name')):
        pool = vocab['ships'] | vocab['items']
    else:
        pool = vocab['items']
    # An empty pool means the vocabulary could not be loaded. Flagging then
    # would mark the entire dataset.
    return bool(pool) and name not in pool


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
        # sto-warp's mirror shards by the first two characters of the sha;
        # the flat path is where it was before that.
        p = local_dir / sha[:2] / f'{sha}.png'
        if not p.exists():
            p = local_dir / f'{sha}.png'
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return img
    from huggingface_hub import hf_hub_download
    for path in crop_paths(sha):
        try:
            local = hf_hub_download(
                repo_id=REPO, repo_type=RTYPE, token=token, filename=path,
            )
        except Exception:
            continue
        img = cv2.imread(local)
        if img is not None:
            return img
    return None



def _scan_weakest(data: dict[str, dict],
                  ledger: dict[str, dict],
                  token: str,
                  local_dir,
                  show_reviewed: bool,
                  limit: int) -> list[dict]:
    """The least-corroborated entries in `data/`, weakest first.

    The other two directions look for a crop whose pixels contradict its
    label. This one looks at nothing but the vote count, because since the
    merge became a queue every entry lands on first sighting and carries its
    own strength: a lone vote is enough to enter `data/` and is recorded as
    exactly that. Most are perfectly good — an item only one person has ever
    confirmed is still an item — but if junk is anywhere, it is here.

    A superseded verdict is shown alongside, so an entry that overturned a
    stronger one is visible as such rather than looking like any other single
    vote.
    """
    # Ranking by vote count alone is useless here: since the merge became a
    # queue almost everything enters on one vote, so the "weakest 200" would
    # be an alphabetical slice of thousands of identical scores. What makes a
    # single vote worth a look is the company it keeps.
    vocab = load_vocabularies()
    scored = []
    for sha, rec in data.items():
        name = (rec.get('name') or '').strip()
        if not name or name in VIRTUAL_LABELS:
            continue
        prior = ledger.get(sha)
        if (prior and prior.get('decision') == 'KEEP'
                and (prior.get('name') or '') == name and not show_reviewed):
            continue
        votes  = int(rec.get('votes') or 0)
        losers = rec.get('losers') or {}
        flags  = []
        # Overturned something better corroborated than itself: the one case
        # where a lone vote is doing real damage if it is wrong.
        if any(int(v or 0) > votes for v in losers.values()):
            flags.append('overturned-stronger')
        # A name nothing downstream can resolve. The models learn it, the
        # exporter cannot write it, and no user typed it on purpose.
        #
        # Which vocabulary counts depends on the slot. A `Ship Type` band
        # holds a ship name and a `Ship Tier` band holds `T6`; neither is an
        # item, so checking them against the item cargo flags every one of
        # them. Measured 2026-09-04 on 12274 entries: 151 flagged, of which
        # 137 were those two slots — 91% noise burying 14 real hits, the same
        # way `no-real-slot` used to.
        if _name_resolves_nowhere(name, rec.get('slot') or '', vocab):
            flags.append('name-not-in-cargo')
        # `slot='migrated'` was a flag here and has been dropped. It fires on
        # 2848 of 12274 entries — a quarter of the dataset — so it ranks
        # nothing, and it buried the two signals that do: an overturned
        # verdict and a name cargo cannot resolve.
        #
        # It is also not the defect it looked like. Measured 2026-09-04: all
        # 2848 are icon-shaped crops, none is a text band, so nothing has
        # leaked into the k-NN pool through the slot check that guards it.
        # The field is unread placeholder metadata from an old import, and
        # 743 of them could not be recovered from the name anyway — a
        # universal console genuinely belongs to four console slots.
        scored.append((-len(flags), votes, sha, rec, prior, flags))
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    scored = scored[:limit]
    n_flagged = sum(1 for t in scored if t[5])
    print(f'Weakest {len(scored)} of {len(data)} entries '
          f'({n_flagged} with something against them) — reading pixels…')

    out: list[dict] = []
    for _rank, votes, sha, rec, prior, flags in scored:
        img = _fetch_crop(sha, token, local_dir=local_dir)
        if img is None:
            continue
        bright, rich = _bright_rich(img)
        losers = rec.get('losers') or {}
        out.append({
            'sha':    sha,
            'name':   (rec.get('name') or '').strip(),
            'slot':   rec.get('slot', ''),
            'bright': bright,
            'rich':   rich,
            'why':    (', '.join(flags) if flags else f'votes={votes}')
                      + (f' (over {", ".join(losers)})' if losers else ''),
            'votes':  votes,
            'img':    img,
            'prior':  (prior or {}).get('decision', ''),
        })
    return out


def scan(snap_dir: Path,
         token: str,
         bright_ratio: float,
         rich_ratio: float,
         show_reviewed: bool,
         direction: str = 'both',
         tail: int = 200) -> list[dict]:
    """Return crops in data/ whose pixels contradict their label.

    Two directions, both reviewed through the same ledger, montage and TSV:

      virtual  a colourful crop labelled `__empty__` / `__inactive__`
      real     a blank cell labelled with an item's name — the mirror, and
               the one that goes on to name every empty slot after that item

    `direction` selects one or both. The mirror direction has to read every
    real-named crop in the dataset, so without sto-warp's local community
    mirror it means thousands of downloads; the caller is warned rather than
    surprised.

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

    if direction == 'tail':
        # Nothing to do with pixel/label contradictions — this one ranks the
        # whole dataset by how well corroborated each entry is.
        return _scan_weakest(data, ledger, token, local_mirror_crops_dir(),
                             show_reviewed, tail)

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
            'why':    'colourful-virtual',
            'img':    img,
            'prior':  (prior or {}).get('decision', '') if prior else '',
        })
    if missing:
        print(f'  WARN: {missing} virtual crop(s) could not be fetched/decoded.')
    out.sort(key=lambda e: e['rich'], reverse=True)

    if direction in ('real', 'both'):
        out += _scan_blank_under_real_name(
            data, ledger, token, local_dir, show_reviewed)
    return out


def _scan_blank_under_real_name(data: dict[str, dict],
                                ledger: dict[str, dict],
                                token: str,
                                local_dir,
                                show_reviewed: bool) -> list[dict]:
    """Crops carrying a real item's name whose pixels read as a blank cell."""
    check = _load_blank_check()
    if check is None:
        return []
    real = [(sha, rec) for sha, rec in data.items()
            if (rec.get('name') or '').strip()
            and (rec.get('name') or '').strip() not in VIRTUAL_LABELS
            and not any((rec.get('slot') or '').startswith(p)
                        for p in _TEXT_CROP_SLOT_PREFIXES)]
    if not local_dir:
        print(f'  NOTE: no local crop mirror — the blank-cell direction would '
              f'download {len(real)} crops. Skipping; run this on a machine '
              f'with the sto-warp community mirror.')
        return []
    print(f'Real-name crops in data/: {len(real)} — checking for blank cells…')

    out: list[dict] = []
    for i, (sha, rec) in enumerate(real, 1):
        if i % 2000 == 0:
            print(f'  … {i}/{len(real)}')
        name = (rec.get('name') or '').strip()
        prior = ledger.get(sha)
        if (prior and prior.get('decision') == 'KEEP'
                and (prior.get('name') or '') == name and not show_reviewed):
            continue
        img = _fetch_crop(sha, token, local_dir=local_dir)
        if img is None or not check(img):
            continue
        bright, rich = _bright_rich(img)
        out.append({
            'sha':    sha,
            'name':   name,
            'slot':   rec.get('slot', ''),
            'bright': bright,
            'rich':   rich,
            'why':    'blank-real',
            'img':    img,
            'prior':  (prior or {}).get('decision', '') if prior else '',
        })
    out.sort(key=lambda e: (e['name'], e['sha']))
    print(f'  blank cells under a real name: {len(out)}')
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
        mark = 'B' if e.get('why') == 'blank-real' else ''
        tag = f'{i + 1}{mark} {e["sha"][:6]}'
        cv2.putText(canvas, tag, (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def write_decisions_tsv(candidates: list[dict], out_path: Path) -> None:
    """Pre-fill a REJECT decision per crop. The maintainer edits the first
    column to KEEP or 'RELABEL <name>' before --apply."""
    lines = ['# decision\tidx\tsha\tlabel\tslot\tbright\trich\twhy',
             '# decision ∈ {REJECT, KEEP, RELABEL <canonical name>}',
             '# why = colourful-virtual (a real icon filed as empty) '
             '| blank-real (an empty cell filed as an item)']
    for i, e in enumerate(candidates, 1):
        lines.append(
            f'REJECT\t{i}\t{e["sha"]}\t{e["name"]}\t{e["slot"]}\t'
            f'{e["bright"]:.3f}\t{e["rich"]:.3f}\t{e.get("why", "")}'
        )
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def read_decisions_tsv(path: Path) -> list[dict]:
    """Parse the edited TSV → [{sha, decision, relabel_name, new_slot}].

    `RELABEL <name>` and `SLOT <slot>` both take an argument after the verb;
    the argument is required, and a verb without one is skipped with a warning
    rather than applied as a bare decision — an empty relabel used to be the
    difference between "fix the name" and "confirm the wrong one".
    """
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
        arg = tok[1].strip() if len(tok) > 1 else ''
        if verb not in ('REJECT', 'KEEP', 'RELABEL', 'SLOT'):
            print(f'  WARN: unknown decision {raw!r} for {sha[:10]} — skipping row')
            continue
        if verb in ('RELABEL', 'SLOT') and not arg:
            print(f'  WARN: {verb} without a value for {sha[:10]} — skipping row')
            continue
        decisions.append({
            'sha': sha,
            'decision': verb,
            'relabel_name': arg if verb == 'RELABEL' else '',
            'new_slot': arg if verb == 'SLOT' else '',
        })
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
    reslot = {s: d.get('new_slot', '') for s, d in by_sha.items()
              if d['decision'] == 'SLOT'}
    keep = {s for s, d in by_sha.items() if d['decision'] == 'KEEP'}

    # Guard: the new slot must be one the dataset already uses, and the
    # record's existing name must be valid under it. The first check catches a
    # typo (`Ship Typ`) without needing a hard-coded slot list to drift out of
    # date; the second catches a slot that is spelled fine and wrong anyway —
    # moving a ship's name to `Fore Weapons` leaves a name no vocabulary for
    # that slot can resolve.
    if reslot:
        known_slots = {(r.get('slot') or '').strip()
                       for r in data.values() if (r.get('slot') or '').strip()}
        vocab = load_vocabularies()
        bad: dict[str, str] = {}
        for s, new_slot in reslot.items():
            rec = data.get(s) or {}
            if new_slot not in known_slots:
                bad[s] = f'{new_slot!r} is not a slot this dataset uses'
            elif _name_resolves_nowhere(rec.get('name') or '', new_slot, vocab):
                bad[s] = (f'{(rec.get("name") or "")!r} does not resolve '
                          f'under {new_slot!r}')
        if bad:
            print('ERROR: SLOT change(s) refused — aborting (no commit):',
                  file=sys.stderr)
            for s, why in bad.items():
                print(f'  {s[:12]}  {why}', file=sys.stderr)
            return False

    # Guard: RELABEL names must exist in the vocabulary the crop's slot
    # allows. Refuse the whole commit otherwise — never write a hand-typed or
    # mistyped label.
    #
    # Per slot, not one global item list: a `Ship Type` crop is labelled with
    # a ship name, which is not an item and would be refused by an item-only
    # check. That is the same mistake `_scan_weakest` made until 2026-09-04,
    # where it flagged every text band as unresolvable.
    if relabel:
        if not canonical:
            print('ERROR: cargo list is empty — cannot validate RELABEL names. '
                  'Point --cargo-dir at a populated sto-warp cache.',
                  file=sys.stderr)
            return False
        vocab = load_vocabularies()
        bad = {s: n for s, n in relabel.items()
               if _name_resolves_nowhere(n, (data.get(s) or {}).get('slot') or '',
                                         vocab)}
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

    # 1. Rewrite data/annotations.jsonl: drop REJECT, apply RELABEL name,
    #    apply SLOT. A SLOT change touches one field — the crop, its name and
    #    its votes all stay, because only the filing was wrong.
    new_data: dict[str, dict] = {}
    for sha, rec in data.items():
        if sha in reject:
            continue
        if sha in relabel:
            rec = dict(rec)
            rec['name'] = relabel[sha]
            rec['relabeled_at'] = _now_iso()
        if sha in reslot:
            rec = dict(rec)
            rec['slot'] = reslot[sha]
            rec['reslotted_at'] = _now_iso()
        new_data[sha] = rec
    ann_payload = ('\n'.join(json.dumps(new_data[s], ensure_ascii=False)
                             for s in sorted(new_data)) + '\n').encode('utf-8')

    ops: list = [CommitOperationAdd(path_in_repo=DATA_ANN,
                                    path_or_fileobj=io.BytesIO(ann_payload))]

    # 2. Delete rejected crop PNGs from data/crops/.
    deleted_crops = 0
    for sha in reject:
        # Either layout — a rejected crop must go wherever it currently sits.
        for dst in crop_paths(sha):
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
    #
    # A SLOT decision must not erase a name pin. The ledger is keyed by sha and
    # the last entry wins, and `democratic_merge_crops._load_relabelled` only
    # honours entries whose `decision` is RELABEL — so writing `decision:
    # 'SLOT'` over a prior RELABEL would quietly unpin the name and let the
    # next client vote overwrite a correction a human had already made. The
    # slot change is recorded as a field alongside the prior verdict instead.
    ledger = _load_ledger(snap_dir)
    for sha, d in by_sha.items():
        prior = ledger.get(sha) or {}
        entry = {
            'crop_sha256': sha,
            'name':        (relabel.get(sha) if sha in relabel
                            else (prior.get('name')
                                  or data.get(sha, {}).get('name') or '')),
            'decision':    d['decision'],
            'reviewed_at': _now_iso(),
        }
        if d['decision'] == 'SLOT':
            entry['slot'] = reslot[sha]
            if prior.get('decision') == 'RELABEL':
                entry['decision'] = 'RELABEL'
        elif prior.get('slot'):
            entry['slot'] = prior['slot']
        ledger[sha] = entry
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
    ap.add_argument('--direction',
                    choices=('virtual', 'real', 'both', 'tail'),
                    default='both',
                    help="What to surface: 'virtual' = a colourful crop "
                         "labelled empty/inactive, 'real' = a blank cell "
                         "labelled with an item name, 'both' = the two "
                         "contradictions (default), 'tail' = the entries "
                         "with the fewest votes behind them, weakest first.")
    ap.add_argument('--tail', type=int, default=200, metavar='N',
                    help='How many of the weakest entries --direction tail '
                         'surfaces (default 200).')
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
                      show_reviewed=args.show_reviewed,
                      direction=args.direction, tail=args.tail)
    if args.direction == 'tail':
        print(f'\n{len(candidates)} least-corroborated entr(ies), weakest '
              f'first — most will be fine; junk, if any, is here:')
    else:
        n_virt = sum(1 for e in candidates if e.get('why') != 'blank-real')
        n_blank = len(candidates) - n_virt
        print(f'\nFlagged {len(candidates)} crop(s) needing review — '
              f'{n_virt} colourful under a virtual label, '
              f'{n_blank} blank under an item name:')
    for i, e in enumerate(candidates, 1):
        prior = f'  (ledger: {e["prior"]})' if e['prior'] else ''
        detail = (f'{e.get("why", "")}'
                  if 'votes' in e
                  else f'why={e.get("why", ""):<17} '
                       f'bright={e["bright"]:.1%} rich={e["rich"]:.1%}')
        print(f'  [{i:>2}] {e["sha"][:10]}  {e["name"]:<34.34} '
              f'slot={e["slot"]!r:<18} {detail}{prior}')

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
