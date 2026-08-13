#!/usr/bin/env python3
"""
democratic_merge_screens.py — screen-type + text-correction merger
====================================================================
Sibling to democratic_merge_crops.py / democratic_merge_anchors.py. Owns
two domains on the icon dataset repo:

    sets-sto/sto-icon-dataset
      staging/<install_id>/screen_types/<TYPE>/<sha>.png   ──┐
      staging/<install_id>/annotations.jsonl  (Ship slots) ──┤  (inputs)
                                                             │
      data/screen_types/<TYPE>/<sha>.png                   ──┐
      data/screen_types/metadata.jsonl                       │  (outputs)
      data/text_corrections.jsonl                          ──┘

Per the audit:

* **D-E.1** — both `data/screen_types/<TYPE>/` and `data/text_corrections.jsonl`
  exist as first-class consensus artefacts (matches the symmetry the
  audit insisted on for every domain).

* **D-E.2 / D-E.5** — voting + drain live in the merger, not the trainer.
  `admin_train.py` is expected to read `data/screen_types/` and
  `data/text_corrections.jsonl` directly (PHASE 3 work).

* **D-E.3** — both artefacts carry `losers`:
    - `data/screen_types/metadata.jsonl` — one line per sha:
        `{"sha": ..., "type": ..., "votes": N, "losers": {type: count}}`
    - `data/text_corrections.jsonl` — one line per ml_name:
        `{"ml_name": ..., "name": ..., "votes": N, "losers": {name: count}}`

* **Z3 (NEW=1, UPDATE=2+)** — both domains apply the asymmetric threshold
  against their respective existing artefacts.

* **D-E.7** — no `__*` filter for either domain (whitelist semantics
  already enforced upstream: `SCREEN_TYPES` for screens, `_is_tier_poison`
  for tier corrections).

Drain policy:
- Promoted screen-type PNGs **are** drained from staging (atomic with the
  data/screen_types/<TYPE>/<sha>.png write).
- Text-correction source lines live inside `staging/<iid>/annotations.jsonl`,
  whose drain is owned by `democratic_merge_crops.py`. This merger only
  reads them. Re-extraction is idempotent — once `data/text_corrections.jsonl`
  carries a correction with N>=threshold votes, future runs see fewer
  source rows but the consensus artefact already remembers the verdict.

Usage:
    python democratic_merge_screens.py               # dry-run
    python democratic_merge_screens.py --apply       # commit to HF
    python democratic_merge_screens.py --apply --min 1
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
UTC = timezone.utc
from pathlib import Path


# ── .env loader ───────────────────────────────────────────────────────────

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

SCREEN_DATA_DIR = 'data/screen_types'
SCREEN_META     = 'data/screen_types/metadata.jsonl'
TEXT_DATA       = 'data/text_corrections.jsonl'

# Screen types must match config/labels.json:screen_types. The list is
# duplicated here intentionally — the merger runs in CI without the
# backend module on its path, and a stale label is filtered out as
# defensive fallback (the upload endpoints already enforce the whitelist).
SCREEN_TYPES = frozenset({
    'SPACE_EQ', 'GROUND_EQ', 'TRAITS', 'SPACE_TRAITS', 'GROUND_TRAITS',
    'BOFFS', 'SPACE_BOFFS', 'GROUND_BOFFS', 'SPECIALIZATIONS',
    'SPACE_MIXED', 'GROUND_MIXED',
    # Skill trees and DISCARD: the client has offered these labels since the
    # skill-tree feature landed, but every upload was refused at the door, so
    # neither could ever accumulate samples. DISCARD is what a screenshot with
    # no build content on it (a doff roster, a loading screen) should be
    # classified as; without the class the model has to force it into one of
    # the build types.
    'SKILLS', 'SPACE_SKILLS', 'GROUND_SKILLS', 'DISCARD',
})

# Text-correction whitelist. Anything outside this slot set is ignored —
# the merger refuses to learn an OCR correction for any slot we don't
# render text for at recognition time.
_TEXT_LEARNING_SLOTS = frozenset({'Ship Type', 'Ship Tier'})

# Closed vocabulary for tier corrections. Used by `_is_tier_poison` to
# kill (canonical_tier -> X) edits at vote-collection time — see
# admin_train.collect_text_corrections for the rationale.
_CANONICAL_TIERS = frozenset({
    'T1', 'T2', 'T3', 'T4', 'T5', 'T5-U', 'T5-X', 'T5-X2',
    'T6', 'T6-X', 'T6-X2',
})


def _is_tier_poison(key: str, val: str) -> bool:
    """A canonical tier on either side of an OCR-correction pair is a
    poison vote (OCR already had it right, or someone is asking us to
    rewrite a perfectly good tier). Matches admin_train.py's filter so
    behaviour does not change with the migration."""
    if key in _CANONICAL_TIERS:
        return True
    if val in _CANONICAL_TIERS and (' ' in key or len(key) > 12):
        return True
    return False


# ── existing-state loaders ────────────────────────────────────────────────

def _load_existing_screens(token: str, repo_files: set[str]) -> dict[str, dict]:
    """Read `data/screen_types/metadata.jsonl` if it exists.

    Returns sha → record {type, votes, losers?}. Missing or empty file
    is fine — first-run path."""
    from huggingface_hub import hf_hub_download
    if SCREEN_META not in repo_files:
        return {}
    try:
        local = hf_hub_download(
            repo_id=REPO, filename=SCREEN_META, repo_type=RTYPE, token=token)
    except Exception as e:
        print(f'NOTICE: failed to read {SCREEN_META} ({e}) — starting fresh')
        return {}
    out: dict[str, dict] = {}
    for line in Path(local).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            sha = (rec.get('sha') or '').strip()
            if sha:
                out[sha] = rec
        except Exception:
            pass
    return out


def _load_existing_text(token: str, repo_files: set[str]) -> dict[str, dict]:
    """Read `data/text_corrections.jsonl`. ml_name → record."""
    from huggingface_hub import hf_hub_download
    if TEXT_DATA not in repo_files:
        return {}
    try:
        local = hf_hub_download(
            repo_id=REPO, filename=TEXT_DATA, repo_type=RTYPE, token=token)
    except Exception as e:
        print(f'NOTICE: failed to read {TEXT_DATA} ({e}) — starting fresh')
        return {}
    out: dict[str, dict] = {}
    for line in Path(local).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            ml  = (rec.get('ml_name') or '').strip()
            if ml:
                out[ml] = rec
        except Exception:
            pass
    return out


# ── vote collection ──────────────────────────────────────────────────────

def _collect_votes(token: str) -> tuple[
    dict[str, Counter],                              # screen_votes[sha] → {type: count}
    dict[str, dict[str, str]],                       # screen_src[sha]   → {iid: staging_path}
    dict[str, Counter],                              # text_votes[ml]    → {name: count}
    int,                                              # rejected_text votes
]:
    """Single shallow clone, two passes — one for screen_types PNGs,
    one for annotations.jsonl text rows."""
    from hf_clone import clone_hf_shallow

    print('Cloning staging tree (shallow)…')
    snap_dir = clone_hf_shallow(REPO, token, repo_type=RTYPE)
    root = Path(snap_dir) / 'staging'
    if not root.exists():
        print(f'WARNING: no staging/ folder at {root}')
        return {}, {}, {}, 0

    # ── (a) Screen-type votes ────────────────────────────────────────────
    screen_votes: dict[str, Counter] = defaultdict(Counter)
    screen_src:   dict[str, dict[str, str]] = defaultdict(dict)

    for png in root.glob('*/screen_types/*/*.png'):
        try:
            install_id = png.parent.parent.parent.name
            stype      = png.parent.name
            sha        = png.stem
        except Exception:
            continue
        if stype not in SCREEN_TYPES:
            continue
        # 1 vote per (install_id, sha). Re-uploads from the same install
        # do not stack — Counter.update is what `defaultdict(Counter)`
        # gives us, but we instead use `[stype] = 1` semantics keyed by
        # iid so a single install gets exactly one say per sha.
        if install_id not in screen_src[sha]:
            screen_src[sha][install_id] = (
                f'staging/{install_id}/screen_types/{stype}/{sha}.png'
            )
            screen_votes[sha][stype] += 1

    # ── (b) Text-correction votes ────────────────────────────────────────
    text_votes:   dict[str, Counter]      = defaultdict(Counter)
    seen_per_install: dict[str, set[tuple[str, str]]] = defaultdict(set)
    rejected = 0

    for af in sorted(root.glob('*/annotations.jsonl')):
        install_id = af.parent.name
        try:
            for line in af.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if (e.get('slot') or '').strip() not in _TEXT_LEARNING_SLOTS:
                    continue
                ml_name = (e.get('ml_name') or '').strip()
                name    = (e.get('name') or '').strip()
                if not (ml_name and name and ml_name != name):
                    continue
                if _is_tier_poison(ml_name, name):
                    rejected += 1
                    continue
                if (ml_name, name) in seen_per_install[install_id]:
                    continue
                seen_per_install[install_id].add((ml_name, name))
                text_votes[ml_name][name] += 1
        except Exception as e:
            print(f'  SKIP text votes from {install_id}: {e}')

    return screen_votes, screen_src, text_votes, rejected


# ── voting ───────────────────────────────────────────────────────────────

def _merge_screens(
    votes:    dict[str, Counter],
    existing: dict[str, dict],
    min_votes:int,
    verbose:  bool,
) -> tuple[dict[str, dict], set[str], list[dict]]:
    """Per-sha majority vote with asymmetric NEW=1 / UPDATE>=min_votes."""
    merged   = dict(existing)
    promoted: set[str] = set()
    report:  list[dict] = []

    for sha, c in sorted(votes.items()):
        winner, count = c.most_common(1)[0]
        old_rec       = existing.get(sha)
        old_type      = (old_rec or {}).get('type', '')
        threshold     = min_votes if sha in existing else 1
        accepted      = count >= threshold

        action = 'SKIP'
        if accepted:
            if old_type == winner:
                action = 'unchanged'
            elif old_type:
                action = 'UPDATE'
            else:
                action = 'NEW'
            losers = {t: v for t, v in c.most_common()[1:4] if t != winner}
            rec: dict = {
                'schema_version': 2,
                'sha':        sha,
                'type':       winner,
                'votes':      count,
                'updated_at': datetime.now(UTC).isoformat(timespec='seconds')
                                              .replace('+00:00', 'Z'),
            }
            if losers:
                rec['losers'] = losers
            merged[sha] = rec
            promoted.add(sha)

        row = {'sha': sha, 'winner': winner, 'votes': count,
               'old_type': old_type, 'action': action}
        report.append(row)
        if verbose or action in ('NEW', 'UPDATE', 'SKIP'):
            print(f'  [screen] {action:<9} {sha[:12]} {winner:<18} '
                  f'votes={count} was={old_type or "-"}')

    return merged, promoted, report


def _merge_text(
    votes:    dict[str, Counter],
    existing: dict[str, dict],
    min_votes:int,
    verbose:  bool,
) -> tuple[dict[str, dict], list[dict]]:
    """Per-ml_name majority vote with asymmetric NEW=1 / UPDATE>=min_votes.

    Re-checks `_is_tier_poison(ml_name, winner)` after the ballot in
    case a 1-vote poison override won (defensive — matches the original
    collect_text_corrections behaviour)."""
    merged  = dict(existing)
    report: list[dict] = []

    for ml, c in sorted(votes.items()):
        winner, count = c.most_common(1)[0]
        if _is_tier_poison(ml, winner):
            report.append({'ml_name': ml, 'winner': winner,
                           'votes': count, 'action': 'SKIP_POISON'})
            if verbose:
                print(f'  [text]   SKIP_POIS {ml!r} → {winner!r} votes={count}')
            continue
        old_rec   = existing.get(ml)
        old_name  = (old_rec or {}).get('name', '')
        threshold = min_votes if ml in existing else 1
        accepted  = count >= threshold

        action = 'SKIP'
        if accepted:
            if old_name == winner:
                action = 'unchanged'
            elif old_name:
                action = 'UPDATE'
            else:
                action = 'NEW'
            losers = {n: v for n, v in c.most_common()[1:4] if n != winner}
            rec: dict = {
                'schema_version': 2,
                'ml_name':    ml,
                'name':       winner,
                'votes':      count,
                'updated_at': datetime.now(UTC).isoformat(timespec='seconds')
                                              .replace('+00:00', 'Z'),
            }
            if losers:
                rec['losers'] = losers
            merged[ml] = rec

        report.append({'ml_name': ml, 'winner': winner, 'votes': count,
                       'old_name': old_name, 'action': action})
        if verbose or action in ('NEW', 'UPDATE', 'SKIP'):
            print(f'  [text]   {action:<9} {ml!r:>22} → {winner!r:<22} '
                  f'votes={count} was={old_name or "-"}')

    return merged, report


# ── apply ────────────────────────────────────────────────────────────────

def _apply(
    api, token: str,
    screens_merged:  dict[str, dict],
    screens_promoted:set[str],
    screen_src:      dict[str, dict[str, str]],
    existing_screens:dict[str, dict],
    text_merged:     dict[str, dict],
    repo_files:      set[str],
):
    """One atomic commit. Updates screen-type artefacts (with drain) +
    text-corrections file. Skipped when nothing changed."""
    from huggingface_hub import (
        CommitOperationAdd, CommitOperationDelete, hf_hub_download,
    )

    ops:    list = []
    new_pngs       = 0
    updated_pngs   = 0
    drained_pngs   = 0

    # 1. data/screen_types/<TYPE>/<sha>.png — copy PNG bytes for each
    #    promoted sha that isn't already at the destination. UPDATE means
    #    the winning type changed: delete the old data/ PNG, write the
    #    new one. NEW just writes.
    for sha in sorted(screens_promoted):
        rec   = screens_merged[sha]
        stype = rec['type']
        dst   = f'{SCREEN_DATA_DIR}/{stype}/{sha}.png'

        old_rec  = existing_screens.get(sha)
        old_type = (old_rec or {}).get('type', '')
        if old_type and old_type != stype:
            old_dst = f'{SCREEN_DATA_DIR}/{old_type}/{sha}.png'
            if old_dst in repo_files:
                ops.append(CommitOperationDelete(path_in_repo=old_dst))
            updated_pngs += 1

        if dst not in repo_files:
            # Need to find a source PNG. Prefer any staging path that
            # actually voted for the WINNING type — otherwise we'd write
            # bytes that vote against the consensus (shouldn't happen
            # because we vote on the sha + bytes are identical, but be
            # defensive).
            src_path = None
            for iid, sp in screen_src.get(sha, {}).items():
                src_path = sp
                break
            if src_path is None:
                # No staging source — sha was promoted on the strength of
                # historical data only. Skip the copy; the metadata row
                # is still emitted so consumers know about the consensus.
                continue
            try:
                local = hf_hub_download(
                    repo_id=REPO, filename=src_path, repo_type=RTYPE, token=token)
            except Exception as e:
                print(f'  ERR fetching {src_path}: {e}')
                continue
            ops.append(CommitOperationAdd(
                path_in_repo    = dst,
                path_or_fileobj = local,
            ))
            new_pngs += 1

    # 2. Drain — every staging PNG path that voted for a promoted sha
    #    gets a Delete op. We delete from EVERY contributor, including
    #    those whose vote lost: their bytes were folded into the same
    #    sha (it's the same image) so they're redundant.
    for sha in screens_promoted:
        for iid, src_path in screen_src.get(sha, {}).items():
            ops.append(CommitOperationDelete(path_in_repo=src_path))
            drained_pngs += 1

    # 3. data/screen_types/metadata.jsonl — full rewrite, sorted by sha
    #    so diffs are reviewable.
    meta_lines = '\n'.join(
        json.dumps(screens_merged[s], ensure_ascii=False)
        for s in sorted(screens_merged)
    )
    meta_bytes = (meta_lines + '\n').encode('utf-8') if meta_lines else b''
    ops.append(CommitOperationAdd(
        path_in_repo    = SCREEN_META,
        path_or_fileobj = io.BytesIO(meta_bytes),
    ))

    # 4. data/text_corrections.jsonl — full rewrite, sorted by ml_name.
    text_lines = '\n'.join(
        json.dumps(text_merged[ml], ensure_ascii=False)
        for ml in sorted(text_merged)
    )
    text_bytes = (text_lines + '\n').encode('utf-8') if text_lines else b''
    ops.append(CommitOperationAdd(
        path_in_repo    = TEXT_DATA,
        path_or_fileobj = io.BytesIO(text_bytes),
    ))

    print(f'Committing: {new_pngs} new screen PNGs, {updated_pngs} retyped, '
          f'drain({drained_pngs} staging PNGs), '
          f'metadata={len(screens_merged)}, text_corrections={len(text_merged)}…')
    api.create_commit(
        repo_id        = REPO,
        repo_type      = RTYPE,
        operations     = ops,
        commit_message = (f'democratic_merge_screens: {len(screens_merged)} screens / '
                          f'{len(text_merged)} corrections '
                          f'(drained {drained_pngs} staging PNGs) '
                          f'@ {datetime.now(UTC).strftime("%Y-%m-%d %H:%M")} UTC'),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description='Democratic merger for screen_types + text_corrections.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--token',   default=HF_TOKEN,
                    help='HF write token (falls back to $HF_TOKEN / .env)')
    ap.add_argument('--apply',   action='store_true',
                    help='Commit to HF (default: dry-run)')
    ap.add_argument('--min',     type=int, default=2, metavar='N',
                    help='Minimum votes to UPDATE existing screen/correction '
                         '(default: 2). NEW entries always need only 1.')
    ap.add_argument('--verbose', action='store_true',
                    help='Print every entry, not only changes')
    args = ap.parse_args()

    if not args.token:
        print('ERROR: HF_TOKEN not set (env, .env, or --token)', file=sys.stderr)
        return 1

    print('=' * 64)
    print(f'Democratic screen/text merger — {REPO}')
    print(f'Min votes: {args.min}  ·  Mode: {"APPLY" if args.apply else "DRY-RUN"}')
    print('=' * 64)

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError
    api = HfApi(token=args.token)

    print('Listing repo files…')
    for attempt in range(5):
        try:
            repo_files = set(api.list_repo_files(repo_id=REPO, repo_type=RTYPE))
            break
        except HfHubHTTPError as e:
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            if status != 429 or attempt == 4:
                raise
            delay = 2 ** attempt
            print(f'  HF 429 on list_repo_files (attempt {attempt + 1}/5) — '
                  f'sleeping {delay}s', file=sys.stderr)
            time.sleep(delay)

    existing_screens = _load_existing_screens(args.token, repo_files)
    existing_text    = _load_existing_text(args.token, repo_files)
    print(f'Existing screens metadata: {len(existing_screens)} sha')
    print(f'Existing text corrections: {len(existing_text)} ml_name')

    screen_votes, screen_src, text_votes, rejected = _collect_votes(args.token)
    print(f'Screen votes: {len(screen_votes)} sha   '
          f'Text votes: {len(text_votes)} ml_name '
          f'({rejected} rejected at ingest)')

    if not screen_votes and not text_votes:
        print('Nothing to merge — exiting.')
        return 0

    screens_merged, screens_promoted, screen_report = _merge_screens(
        screen_votes, existing_screens, args.min, args.verbose)

    text_merged, text_report = _merge_text(
        text_votes, existing_text, args.min, args.verbose)

    sc_new    = sum(1 for r in screen_report if r['action'] == 'NEW')
    sc_upd    = sum(1 for r in screen_report if r['action'] == 'UPDATE')
    sc_skip   = sum(1 for r in screen_report if r['action'] == 'SKIP')
    tx_new    = sum(1 for r in text_report   if r['action'] == 'NEW')
    tx_upd    = sum(1 for r in text_report   if r['action'] == 'UPDATE')
    tx_skip   = sum(1 for r in text_report   if r['action'] == 'SKIP')
    tx_poison = sum(1 for r in text_report   if r['action'] == 'SKIP_POISON')

    print()
    print('─── Summary ──────────────────────────────────────────')
    print(f'  Screens   + NEW: {sc_new}   ~ UPDATE: {sc_upd}   - SKIP: {sc_skip}')
    print(f'  Text      + NEW: {tx_new}   ~ UPDATE: {tx_upd}   - SKIP: {tx_skip}   '
          f'poison: {tx_poison}')
    print(f'  Totals after merge — screens: {len(screens_merged)}, '
          f'text: {len(text_merged)}')

    # Uniform drain monitor (post-audit TODO #5).
    # Screens domain promotes by sha (PNG drain). Text domain has no PNG drain
    # — it shares annotations.jsonl with crops, so promoted=0 there.
    print(f'DRAIN: domain=screens promoted={len(screens_promoted)} '
          f'new={sc_new} update={sc_upd} skip={sc_skip}')
    print(f'DRAIN: domain=text promoted=0 '
          f'new={tx_new} update={tx_upd} skip={tx_skip} poison={tx_poison}')

    if not args.apply:
        print('\nDRY-RUN — use --apply to commit.')
        return 0

    if not screens_promoted and screens_merged == existing_screens \
       and text_merged == existing_text:
        print('Nothing actually changed — no commit.')
        return 0

    _apply(api, args.token,
           screens_merged, screens_promoted, screen_src, existing_screens,
           text_merged, repo_files)
    print('OK — committed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
