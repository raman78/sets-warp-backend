#!/usr/bin/env python3
"""
admin_clean_labels.py — Repair corrupt labels in HF staging annotations
========================================================================
One-shot cleanup tool. Scans every staging/<install_id>/annotations.jsonl on
the training-crops HF dataset, validates each entry's `name` against the
canonical SETS cache (equipment + boff abilities + traits + starship traits),
and rewrites entries whose `name` is recoverable. Drops entries that cannot
be matched to any canonical name.

Why: early WARP CORE versions allowed free-text item names. Some contributor
also bulk-uploaded crops with labels reverse-engineered from filenames —
filename `safe_name` is sanitized as `name.replace(' ', '_').lower()[:40]`,
and the reverse pipeline (`'_' -> ' '`, `[word.capitalize() for word in ...]`)
produced pretty-cased BUT 40-char-truncated and acronym-broken names
(`K.h.g.` instead of `K.H.G.`, `Adapted K.h.g. Disruptor Battle Rifle MK`
instead of `Adapted K.H.G. Disruptor Battle Rifle Mk XII`).

Effect on training: label_map gets 700+ truncated classes + 47 case-only
duplicate pairs (`A Good Day To Die` vs `A Good Day to Die`), which split
the gallery and weaken ArcFace recall.

After cleanup → re-run `train_metric_model.yml` for a clean label_map.

Run from sets-warp dir:
    .venv/bin/python ../sets-warp-backend/admin_clean_labels.py
    .venv/bin/python ../sets-warp-backend/admin_clean_labels.py --apply

Environment variables (.env, same as admin_train.py):
    HF_TOKEN, HF_DATASET (default: sets-sto/sto-icon-dataset)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Upgrade marker: ` Mk I` / ` MK XII` / ` Mk V` etc. — anything after the
# mark token is item-quality metadata, never part of the canonical name.
_MK_SUFFIX_RE = re.compile(r'\s+M[Kk]\s+[IVX]+\b.*$')

# Reuse HF helpers + .env loader from admin_train.py
sys.path.insert(0, str(Path(__file__).parent))
from admin_train import (  # noqa: E402
    HF_TOKEN, HF_DATASET,
    _require_hf, _list_staging_folders, _load_staging_annotations,
    _create_commit_with_retry,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

UTC = timezone.utc

# Slots whose annotations are NOT for icon classification — leave their
# `name` field untouched (Ship Type / Ship Tier feed OCR correction training,
# not the icon classifier — see collect_text_corrections() in admin_train.py).
TEXT_LEARNING_SLOTS = {'Ship Type', 'Ship Tier'}

# Virtual class names that bypass canonical validation.
VIRTUAL_NAMES = {'__empty__', '__inactive__'}

# Captain specialization names — hardcoded in WARP CORE (not in cargo).
# Source: warp/trainer/trainer_window.py:SPECIALIZATION_NAMES. Keep in sync.
SPECIALIZATION_NAMES = {
    'Command Officer', 'Intelligence Officer', 'Miracle Worker', 'Pilot',
    'Temporal Operative', 'Constable', 'Commando', 'Strategist',
}


# ── Canonical name index ──────────────────────────────────────────────────────

def _sanitize_equipment_name(name: str) -> str:
    """Mirror of src/textedit.py:sanitize_equipment_name() — same cleanup the
    SETS client applies before storing equipment cache keys, so our canonical
    set matches what users actually see in the dropdown."""
    name = name.replace('&quot;', '"').replace('&#34;', '"')
    if '∞' in name:
        name, _ = name.split('∞', 1)
    # Strip any Mk + roman numeral suffix (Mk I … Mk XV — full upgrade range).
    name = _MK_SUFFIX_RE.sub('', name)
    if '[' in name:
        name, _ = name.split('[', 1)
    if name[-2:] == '-S':
        name = name[:-2]
    return name.strip()


def build_canonical(cargo_dir: Path) -> set[str]:
    """Build canonical set of item / trait / ability names from local cargo
    JSON files (same files SETS would load at startup).
    """
    canonical: set[str] = set()

    # 1. Equipment (one flat file with type per row)
    eq_path = cargo_dir / 'equipment.json'
    if eq_path.exists():
        try:
            for it in json.loads(eq_path.read_text(encoding='utf-8')):
                n = it.get('name')
                if isinstance(n, str) and n:
                    canonical.add(_sanitize_equipment_name(n))
        except Exception as e:
            log.warning(f'equipment.json: {e}')
    else:
        log.warning(f'equipment.json not found at {eq_path}')

    # 2. Boff abilities — `all` is flat {name: tooltip}
    boff_path = cargo_dir / 'boff_abilities.json'
    if boff_path.exists():
        try:
            d = json.loads(boff_path.read_text(encoding='utf-8'))
            canonical.update(d.get('all', {}).keys())
        except Exception as e:
            log.warning(f'boff_abilities.json: {e}')

    # 3. Traits — flat list of {name, environment, type, ...} dicts
    traits_path = cargo_dir / 'traits.json'
    if traits_path.exists():
        try:
            for it in json.loads(traits_path.read_text(encoding='utf-8')):
                n = it.get('name')
                if isinstance(n, str) and n:
                    canonical.add(n)
        except Exception as e:
            log.warning(f'traits.json: {e}')

    # 4. Starship traits — flat list of {name, ...} dicts
    st_path = cargo_dir / 'starship_traits.json'
    if st_path.exists():
        try:
            for it in json.loads(st_path.read_text(encoding='utf-8')):
                n = it.get('name')
                if isinstance(n, str) and n:
                    canonical.add(n)
        except Exception as e:
            log.warning(f'starship_traits.json: {e}')

    # 5. Captain specializations — not in cargo; hardcoded in WARP CORE.
    canonical.update(SPECIALIZATION_NAMES)

    # Strip empties
    canonical.discard('')
    return canonical


# ── Matching ──────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Match-key: lowercase, collapse whitespace, drop punctuation that drifts
    between user input and cargo (period after acronym letters etc.).
    Used only for lookup — never written back to annotations."""
    return ' '.join(s.lower().replace('.', '').replace('-', ' ').split())


def build_lookup(canonical: set[str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Return (exact_norm_lookup, prefix_lookup).

    exact_norm_lookup: {normalized → canonical}.
      One key may have multiple canonical names if cargo itself has
      hyphen/period variants — last write wins (rare; logged separately).
    prefix_lookup: {first-40-norm-chars → [canonical, ...]}.
      Used to recover 40-char-truncated names by prefix.
    """
    exact: dict[str, str] = {}
    prefix: dict[str, list[str]] = defaultdict(list)
    for c in canonical:
        n = _norm(c)
        if not n:
            continue
        exact[n] = c
        # Truncation in the legacy filename pipeline was: replace ' '→'_',
        # lowercase, then `[:40]`. After our normalization (drop '.', '-' → ' ')
        # the 40-char window covers slightly different chars, so use a
        # forgiving prefix length: first 30 normalized chars is enough to
        # disambiguate most STO items.
        prefix[n[:30]].append(c)
    return exact, prefix


def repair_name(name: str,
                canonical: set[str],
                exact: dict[str, str],
                prefix: dict[str, list[str]]) -> tuple[str | None, str]:
    """Return (repaired_name, action). `action` ∈ {keep, case, truncated, drop}.
    repaired_name=None means drop.
    """
    if not name:
        return None, 'drop'
    if name in canonical:
        return name, 'keep'

    n = _norm(name)

    # Case / punctuation difference only
    if n in exact:
        return exact[n], 'case'

    # Sanitize input the same way cargo entries were sanitized (strip Mk X /
    # MK X suffix, [...] mods, trailing -S, &quot;) — covers annotations that
    # still carry mark/quality suffixes the canonical set discards.
    sanitized = _sanitize_equipment_name(name)
    if sanitized and sanitized != name:
        if sanitized in canonical:
            return sanitized, 'case'
        ns = _norm(sanitized)
        if ns in exact:
            return exact[ns], 'case'

    # 40-char truncation: lookup by 30-char normalized prefix.
    # The legacy filename truncation happened BEFORE the reverse-capitalize,
    # so anything ≥38 chars in the original is suspect; require unique match.
    # Two directions: canonical may be longer (true truncation) OR the
    # annotation may carry trailing noise like 'MK' / 'Mk Xii' that the
    # canonical doesn't have (also unique-match-required).
    if len(name) >= 38:
        candidates = prefix.get(n[:30], [])
        viable = [c for c in candidates
                  if _norm(c).startswith(n) or n.startswith(_norm(c))]
        if len(viable) == 1:
            return viable[0], 'truncated'

    return None, 'drop'


# ── HF write ──────────────────────────────────────────────────────────────────

def write_cleaned_jsonl(install_id: str, cleaned_entries: list[dict], api) -> bool:
    """Replace staging/<install_id>/annotations.jsonl with `cleaned_entries`.
    One commit per install_id so individual contributors can be rolled back
    without affecting others."""
    from huggingface_hub import CommitOperationAdd
    import io

    buf = io.StringIO()
    for e in cleaned_entries:
        buf.write(json.dumps(e, ensure_ascii=False))
        buf.write('\n')
    blob = buf.getvalue().encode('utf-8')

    op = CommitOperationAdd(
        path_in_repo=f'staging/{install_id}/annotations.jsonl',
        path_or_fileobj=blob,
    )
    commit_msg = f'admin_clean_labels: repair {install_id} annotations'
    return _create_commit_with_retry(api, HF_DATASET, 'dataset', [op], commit_msg)


def _build_jsonl_op(install_id: str, cleaned_entries: list[dict]):
    """Build a CommitOperationAdd for a single install_id's cleaned JSONL."""
    from huggingface_hub import CommitOperationAdd
    import io
    buf = io.StringIO()
    for e in cleaned_entries:
        buf.write(json.dumps(e, ensure_ascii=False))
        buf.write('\n')
    blob = buf.getvalue().encode('utf-8')
    return CommitOperationAdd(
        path_in_repo=f'staging/{install_id}/annotations.jsonl',
        path_or_fileobj=blob,
    )


def write_cleaned_batch(batch: list[tuple[str, list[dict]]], api) -> bool:
    """Write a batch of install_ids in ONE HF commit.
    Batching is the recommended way to stay under HF's 128 commits/hour limit.
    Trade-off: rollback granularity is per-batch, not per-install_id (backups
    on local disk remain per-install_id so manual recovery is still possible).
    """
    ops = [_build_jsonl_op(iid, cleaned) for iid, cleaned in batch]
    ids_preview = ', '.join(iid[:8] for iid, _ in batch[:3])
    if len(batch) > 3:
        ids_preview += f', +{len(batch) - 3} more'
    commit_msg = f'admin_clean_labels: repair {len(batch)} install_ids ({ids_preview})'
    return _create_commit_with_retry(api, HF_DATASET, 'dataset', ops, commit_msg)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Validate / repair labels in HF staging annotations.jsonl',
    )
    parser.add_argument(
        '--cargo-dir',
        type=str,
        default=str(Path(__file__).parent.parent / 'sets-warp' / 'data' / 'cargo'),
        help='Directory with SETS cargo JSON files (equipment.json, '
             'boff_abilities.json, traits.json, starship_traits.json).',
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Write cleaned annotations back to HF (default: dry-run).',
    )
    parser.add_argument(
        '--backup-dir',
        type=str,
        default=None,
        help='If set, save each install_id\'s ORIGINAL annotations.jsonl '
             'to <backup-dir>/<install_id>.jsonl before writing.',
    )
    parser.add_argument(
        '--only-install-id',
        type=str,
        default=None,
        help='Process only this install_id (for testing).',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='Number of install_ids per HF commit (default: 20). '
             'Set to 1 to revert to per-install commits. HF rate limit is '
             '128 commits/hour shared with WARP CORE sync, so default 20 '
             'keeps the full run under ~40 commits.',
    )
    parser.add_argument(
        '--skip-existing-backups',
        action='store_true',
        help='Skip install_ids that already have a backup file in '
             '--backup-dir (use to resume an interrupted run).',
    )
    args = parser.parse_args()

    cargo_dir = Path(args.cargo_dir)
    if not cargo_dir.is_dir():
        print(f'ERROR: cargo-dir does not exist: {cargo_dir}', file=sys.stderr)
        sys.exit(2)

    print('=' * 64)
    print('WARP Label Cleanup')
    print(f'Cargo:   {cargo_dir}')
    print(f'Mode:    {"APPLY" if args.apply else "DRY-RUN"}')
    print('=' * 64)

    canonical = build_canonical(cargo_dir)
    print(f'\nCanonical names: {len(canonical)}')
    if len(canonical) < 100:
        print('ERROR: canonical set looks too small — check --cargo-dir', file=sys.stderr)
        sys.exit(2)
    exact, prefix = build_lookup(canonical)
    print(f'Normalized keys: {len(exact)}, prefix buckets: {len(prefix)}')

    _require_hf()
    print('\nListing staging folders...')
    folders = _list_staging_folders()
    if args.only_install_id:
        folders = [f for f in folders if f == args.only_install_id]
    if not folders:
        print('No staging folders.')
        return
    print(f'  {len(folders)} install_id(s)')

    backup_dir = Path(args.backup_dir) if args.backup_dir else None
    if backup_dir:
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f'  Backup → {backup_dir}')

    api = None
    if args.apply:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)

    # ── Pass 1: scan + classify ────────────────────────────────────────────
    total_in = 0
    total_keep = total_case = total_trunc = total_drop = total_skip = 0
    drop_samples: list[tuple[str, str, str]] = []  # (install_id, slot, name)
    case_samples: list[tuple[str, str]] = []      # (old, new)
    trunc_samples: list[tuple[str, str]] = []
    by_install: dict[str, list[dict]] = {}

    for i, iid in enumerate(folders, 1):
        if i % 10 == 0:
            print(f'  ... {i}/{len(folders)}')
        anns = _load_staging_annotations(iid)
        if not anns:
            continue

        cleaned: list[dict] = []
        for entry in anns:
            total_in += 1
            slot = entry.get('slot', '')
            name = entry.get('name', '').strip()

            if slot in TEXT_LEARNING_SLOTS or name in VIRTUAL_NAMES:
                cleaned.append(entry)
                total_skip += 1
                continue

            repaired, action = repair_name(name, canonical, exact, prefix)
            if action == 'keep':
                cleaned.append(entry)
                total_keep += 1
            elif action == 'case':
                new_entry = dict(entry)
                new_entry['name'] = repaired
                cleaned.append(new_entry)
                total_case += 1
                if len(case_samples) < 15:
                    case_samples.append((name, repaired))
            elif action == 'truncated':
                new_entry = dict(entry)
                new_entry['name'] = repaired
                cleaned.append(new_entry)
                total_trunc += 1
                if len(trunc_samples) < 15:
                    trunc_samples.append((name, repaired))
            else:  # drop
                total_drop += 1
                if len(drop_samples) < 15:
                    drop_samples.append((iid, slot, name))

        by_install[iid] = cleaned

    # ── Report ─────────────────────────────────────────────────────────────
    print('\n── Scan complete ──')
    print(f'  Total entries seen: {total_in}')
    print(f'  Kept as-is:         {total_keep}')
    print(f'  Skipped (text/virt):{total_skip}')
    print(f'  Case-fixed:         {total_case}')
    print(f'  Truncation-fixed:   {total_trunc}')
    print(f'  Dropped:            {total_drop}')

    if case_samples:
        print('\nCase-fix samples:')
        for o, n in case_samples:
            print(f'  {o!r}  →  {n!r}')
    if trunc_samples:
        print('\nTruncation-fix samples:')
        for o, n in trunc_samples:
            print(f'  {o!r}  →  {n!r}')
    if drop_samples:
        print('\nDrop samples (unrecoverable):')
        for iid, slot, name in drop_samples:
            print(f'  [{iid[:8]}…][{slot}] {name!r}')

    # ── Pass 2: write back ────────────────────────────────────────────────
    if not args.apply:
        print('\nDry-run finished. Re-run with --apply to write back to HF.')
        return

    # Skip install_ids already backed up (resume support).
    if args.skip_existing_backups and backup_dir:
        before = len(by_install)
        by_install = {
            iid: cleaned for iid, cleaned in by_install.items()
            if not (backup_dir / f'{iid}.jsonl').exists()
        }
        print(f'\nSkipping {before - len(by_install)} already-backed-up '
              f'install_ids; {len(by_install)} remain.')

    batch_size = max(1, args.batch_size)
    n_batches = (len(by_install) + batch_size - 1) // batch_size
    print(f'\nWriting cleaned annotations to HF '
          f'(batch_size={batch_size}, {n_batches} commit(s))...')

    items = list(by_install.items())
    n_written_ids = n_failed_ids = n_commits_ok = n_commits_failed = 0
    for batch_idx in range(n_batches):
        batch = items[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        # Optional backup of originals (one file per install_id, regardless
        # of batching — granularity for manual recovery stays per-contributor).
        if backup_dir:
            for iid, _ in batch:
                bpath = backup_dir / f'{iid}.jsonl'
                if bpath.exists():
                    continue
                orig = _load_staging_annotations(iid)
                with bpath.open('w', encoding='utf-8') as f:
                    for e in orig:
                        f.write(json.dumps(e, ensure_ascii=False) + '\n')

        ok = write_cleaned_batch(batch, api)
        if ok:
            n_commits_ok += 1
            n_written_ids += len(batch)
        else:
            n_commits_failed += 1
            n_failed_ids += len(batch)
        print(f'  batch {batch_idx + 1}/{n_batches}: '
              f'{"OK" if ok else "FAIL"} ({len(batch)} install_ids)')
        # Spread commits across the hour. With batch_size=20 and 5-sec sleep,
        # ~12 commits/min × 60s ≈ 720 → safely under the 128/h API window
        # even when WARP CORE sync is competing.
        time.sleep(5.0)

    print(f'\nDone. {n_commits_ok}/{n_batches} commits written '
          f'({n_written_ids} install_ids); {n_commits_failed} failed '
          f'({n_failed_ids} install_ids).')
    if n_commits_failed:
        print('Re-run with --apply --skip-existing-backups to retry '
              '(already-backed-up install_ids will be skipped).')


if __name__ == '__main__':
    main()
