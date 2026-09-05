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
    - no threshold: staging is a queue, so tallying settles every entry and
      drains it. Votes express confidence *in* a record rather than gating
      entry to it — they accumulate on agreement, and a superseded verdict
      keeps its strength so an overturn is auditable. `--min` survives for
      `admin_merge.py`, which still gates knowledge entries on it.
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

# Maintainer review ledger written by admin_reject_crops.py. Shas decided
# REJECT there must never be re-promoted, even if a user re-uploads the same
# colourful crop that was mislabeled __empty__/__inactive__.
LEDGER   = 'data/reviewed_virtual.jsonl'


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


def _load_rejected_shas(snap_dir) -> set[str]:
    """Return the set of crop_sha256 the maintainer marked REJECT in the
    review ledger (data/reviewed_virtual.jsonl). These are permanently barred
    from data/ — re-uploads must not resurrect them. Missing ledger = empty."""
    out: set[str] = set()
    local = Path(snap_dir) / LEDGER
    if not local.exists():
        return out
    with open(local, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get('decision') == 'REJECT' and d.get('crop_sha256'):
                out.add(d['crop_sha256'])
    return out


def _load_relabelled(snap_dir) -> dict[str, str]:
    """crop_sha256 → the name the maintainer relabelled it to.

    The ledger records RELABEL beside REJECT, and until now only REJECT was
    read back. A relabel was therefore written to `data/` and left unguarded:
    the merge is a queue, so the next client to upload that crop under its old
    name overwrote the correction with a single vote. Measured 2026-09-04 —
    `Fleet Support Cruiser (T6)`, corrected at 10:24, was back by 16:28.

    A maintainer looked at the picture; a client's label is whatever its
    recogniser or its user offered. So the maintainer's name is pinned and
    incoming votes cannot move it. Last entry wins, which is what makes a
    correction correctable — re-reviewing the same sha supersedes it.
    """
    out: dict[str, str] = {}
    local = Path(snap_dir) / LEDGER
    if not local.exists():
        return out
    with open(local, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if (d.get('decision') == 'RELABEL' and d.get('crop_sha256')
                    and (d.get('name') or '').strip()):
                out[d['crop_sha256']] = d['name'].strip()
    return out


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
    rejected: set[str] | None = None,
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
      - name_votes[sha][slot][name] → count of distinct install_ids voting for
        that name **on that slot**. The slot is part of the key because a name
        is only an answer to "what is in this slot?" — see `_merge`.
      - slot_votes[sha][slot]   → count of distinct install_ids voting for slot
      - crop_src[sha]           → first staging path that has this crop PNG
      - per_install[install_id] → number of entries contributed by that install
      - contributors_for_sha[sha] → install_ids whose annotations voted on sha
        (used by drain — delete staging/<iid>/crops/<sha>.png after promotion)
      - staging_records[install_id] → raw annotation dicts kept for the
        staging rewrite (drain trims entries whose sha was promoted)
    """
    rejected = rejected or set()
    root = Path(snap_dir) / 'staging'
    if not root.exists():
        print(f'WARNING: no staging/ folder at {root}')
        return {}, {}, {}, {}, {}, {}

    anno_files = sorted(root.glob('*/annotations.jsonl'))
    print(f'Found {len(anno_files)} contributors with annotations.')

    name_votes: dict[str, dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter))
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
                    if sha in rejected:
                        # Maintainer-rejected crop re-uploaded: never re-promote.
                        continue
                    # Dedup duplicate uploads from one install: one vote each.
                    key = (sha, name, slot)
                    if key in seen_in_install:
                        continue
                    seen_in_install.add(key)
                    name_votes[sha][slot][name] += 1
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


def _ballot_for(sha_votes: dict[str, Counter], slot: str) -> tuple[Counter, dict[str, Counter]]:
    """Split one crop's name votes into the ballot that counts and the rest.

    A name is not an answer on its own — it answers *"what is in this slot?"*.
    Counting every name for a crop in one ballot regardless of slot is what
    produced records like `name='Fleet Yamaguchi Support Cruiser'` with
    `slot='Ship Tier'`: the name came from the class-line ballot, the slot from
    the tier ballot, and the pair had never been submitted by anybody. It
    happens whenever two rows share a bounding box, because identical pixels
    give an identical hash — the ship class line and the tier badge did exactly
    that when no badge was found on screen.

    Entries carrying no slot at all are folded into the winner rather than
    dropped: they are not asserting a *different* slot, they are simply not
    saying, so they still count toward whatever the crop turns out to be.

    Returns (ballot, dropped_by_slot) so the caller can report what it set
    aside instead of discarding it in silence.
    """
    ballot = Counter(sha_votes.get(slot) or Counter())
    if slot:
        ballot.update(sha_votes.get('') or Counter())
    dropped = {s: c for s, c in sha_votes.items()
               if s != slot and s != '' and c}
    return ballot, dropped


def _merge(
    name_votes: dict[str, dict[str, Counter]],
    slot_votes: dict[str, Counter],
    existing:   dict[str, dict],
    verbose:    bool,
    rejected:   set[str] | None = None,
    relabelled: dict[str, str] | None = None,
) -> tuple[dict[str, dict], list[dict], set[str]]:
    """Majority vote. Returns (merged, report_rows, promoted_shas).

    `promoted_shas` is the set this run actually accepted (NEW + UPDATE +
    unchanged). The drain in `_apply` uses this set so staging is cleaned
    even when the consensus was already reflected in data/ — those votes
    have done their job and should not be re-tallied next run.
    """
    # Drop legacy poison entries + maintainer-rejected shas from existing
    # (one-shot self-heal — keeps data/ clean even if one slipped in).
    rejected = rejected or set()
    merged = {sha: rec for sha, rec in existing.items()
              if not _is_poison_name((rec.get('name') or ''))
              and sha not in rejected}
    dropped_poison = len(existing) - len(merged)
    if dropped_poison:
        print(f'[clean] dropped {dropped_poison} legacy poison / rejected entries')

    # Same self-heal for a correction that was overwritten before the pin
    # existed. Without this the pin only bites on the *next* upload of that
    # crop, so a name overwritten in the past stays wrong until someone
    # happens to photograph the same slot again — which for a rare item is
    # never. Applied here so one merge run restores every past correction.
    healed = 0
    for sha, name in (relabelled or {}).items():
        rec = merged.get(sha)
        if rec is not None and (rec.get('name') or '') != name:
            rec = dict(rec)
            rec['name'] = name
            rec['relabeled_at'] = datetime.now(UTC).isoformat(
                timespec='seconds').replace('+00:00', 'Z')
            merged[sha] = rec
            healed += 1
    if healed:
        print(f'[clean] restored {healed} maintainer correction(s) that had '
              f'been overwritten')

    report: list[dict] = []
    promoted_shas: set[str] = set()

    pinned = relabelled or {}

    for sha, by_slot in sorted(name_votes.items()):
        # The slot is decided first, and the name is then decided *within* it.
        # Both halves of the record therefore come from the same voters, so the
        # published pair is one that people actually submitted.
        slot_c = slot_votes.get(sha) or Counter()
        slot   = slot_c.most_common(1)[0][0] if slot_c else ''
        votes, dropped_by_slot = _ballot_for(by_slot, slot)
        if not votes:
            continue
        if dropped_by_slot:
            # Not discarded quietly: a crop claimed by two slots is a client
            # sending one picture for two rows, and the count is how anyone
            # finds out it is still happening.
            detail = ', '.join(
                f'{s}={sum(c.values())}' for s, c in sorted(dropped_by_slot.items()))
            print(f'  [slot] {sha[:12]} settled as {slot!r}; '
                  f'not counted here: {detail}')
        winner, count = votes.most_common(1)[0]
        # A maintainer who reviewed this crop outranks the tally. They looked
        # at the picture; a client's label is whatever its recogniser or its
        # user offered, and one such vote used to overwrite the correction on
        # the next merge. The losing votes are still recorded below, so the
        # disagreement stays visible rather than being erased.
        if sha in pinned and pinned[sha] != winner:
            votes = Counter(votes)
            votes[pinned[sha]] = max(votes.values()) + 1
            winner, count = pinned[sha], votes[pinned[sha]]
        old_rec       = existing.get(sha)
        old_name      = (old_rec or {}).get('name', '')

        # Staging is a queue, not a holding pen: an entry has arrived from a
        # client, has not been tallied yet, and is not in the models. Tallying
        # it settles it either way, so every entry is applied and staging
        # empties. Votes then express confidence in the record rather than
        # gating entry to it — they accumulate on agreement and reset when a
        # verdict is superseded, so a weak signal is visibly weak and can be
        # reviewed as such.
        #
        # The bar this replaces was "a second, independent voice to overturn",
        # which is sound for a crowd and means "never" for this project:
        # measured 2026-09-03 there are two contributors with annotations, and
        # 102 corrections had been waiting indefinitely — among them a crop
        # whose stored name, `Attack Pattern Beta'`, no cargo row has ever
        # matched. Nothing surfaced them, and the models kept training on the
        # label a human had already corrected.
        accepted = True

        old_votes = int((old_rec or {}).get('votes') or 0)

        action = 'SKIP'
        if accepted:
            if old_name == winner:
                action = 'unchanged'
            elif old_name:
                action = 'UPDATE'
            else:
                action = 'NEW'

            # `slot` was decided at the top of the loop, before the name, and
            # the name ballot was drawn from that slot's voters only.
            losers_dict = {n: v for n, v in votes.most_common()[1:4] if n != winner}
            if action == 'unchanged':
                # Agreement accumulates. Replacing the count with this
                # batch's would make five confirmations read as one, which
                # is the opposite of what the number is for.
                total_votes = old_votes + count
                losers_dict = {**((old_rec or {}).get('losers') or {}),
                               **losers_dict}
            else:
                total_votes = count
            if action == 'UPDATE':
                # The superseded verdict keeps its strength on the record, so
                # an overturn is auditable and reversible rather than a
                # silent replacement.
                losers_dict = {old_name: old_votes or 1, **losers_dict}
            entry: dict = {
                'schema_version': 2,
                'crop_sha256': sha,
                'name':        winner,
                'slot':        slot,
                'votes':       total_votes,
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


# ── Commit safety ───────────────────────────────────────────────────────────

from hf_commit import CHUNK as COMMIT_CHUNK, commit_chunked, validate_ops


def _commit_in_stages(api, crop_ops: list, anno_op, drain_ops: list,
                      summary: str, chunk: int = COMMIT_CHUNK) -> None:
    """Apply the merge as an ordered sequence of commits.

    The order is what keeps every intermediate state consistent:

      1. crop PNGs       additive. A crop in `data/crops/` that no annotation
                         references yet is inert, and the next run skips it
                         as already present.
      2. annotations     every path it references now exists.
      3. staging drain   staging is only emptied once `data/` is
                         authoritative, so an interrupted run leaves
                         duplicates to re-promote, never a lost crop.

    Stopping anywhere in that sequence is safe and the next scheduled run
    converges: promotion is idempotent by sha. See `hf_commit` for why the
    single atomic commit this replaced had to go.
    """
    problems = validate_ops(crop_ops + [anno_op] + drain_ops)
    if problems:
        for p in problems[:20]:
            print(f'  MALFORMED: {p}')
        raise SystemExit(
            f'Refusing to commit: {len(problems)} malformed operation(s). '
            f'Nothing was written.')

    for ops, label in ((crop_ops, 'crops'), ([anno_op], 'annotations'),
                       (drain_ops, 'drain')):
        commit_chunked(api, REPO, RTYPE, ops, summary,
                       chunk=chunk, label=label, validate=False)


def _surviving_rows(
    records:       dict[str, list[dict]],
    staged_shas:   set[str],
    existing:      dict[str, dict],
    safe_promoted: set[str],
    barred:        set[str],
) -> dict[str, list[dict]]:
    """The staging rows worth keeping, per install.

    A row is dropped when the tally can never act on it again. Four ways that
    happens, and only the first was handled before:

    * promoted this run       — it has done its job and is drained
    * crop exists nowhere     — tallying reads the PNG's bytes, so a row with
                                no PNG in staging and none in `data/` can
                                never be promoted. The mirror of the orphan
                                PNG swept further down, and the direction that
                                had no sweep.
    * barred by the ledger    — the maintainer rejected the crop. The tally
                                skips it, so nothing else would remove its
                                staging copy: the rejection would be
                                re-litigated on every run, for good.
    * no sha at all           — a reference to nothing.

    Everything else survives, including an ordinary single vote waiting for
    company. Dropping one of those would silently discard a contribution,
    which is worse than any residue this sweeps.
    """
    out: dict[str, list[dict]] = {}
    for iid, rows in records.items():
        kept = []
        for rec in rows:
            sha = (rec.get('crop_sha256') or '').strip()
            if not sha or sha in barred or sha in safe_promoted:
                continue
            if sha in staged_shas or sha in existing:
                kept.append(rec)
        out[iid] = kept
    return out


def _apply(
    api, token: str,
    merged:    dict[str, dict],
    promoted_shas: set[str],
    existing:  dict[str, dict],
    crop_src:  dict[str, str],
    repo_files: set[str],
    contributors_for_sha: dict[str, set[str]],
    staging_records: dict[str, list[dict]],
    rejected: set[str] | None = None,
    chunk: int = COMMIT_CHUNK,
):
    """Rewrite data/annotations.jsonl, copy approved crops, then drain
    staging — delete staging crop PNGs for promoted sha and rewrite each
    contributor's annotations.jsonl keeping only the not-promoted lines.

    Applied as an ordered sequence of commits rather than one; see
    `_commit_in_stages` for the order and why a half-applied state is still
    safe.
    """
    from huggingface_hub import (
        CommitOperationAdd, CommitOperationDelete, hf_hub_download,
    )

    # 1. Annotations file (one JSON object per line, sorted by sha for diffability).
    lines: list[str] = []
    for sha in sorted(merged):
        lines.append(json.dumps(merged[sha], ensure_ascii=False))
    payload = ('\n'.join(lines) + '\n').encode('utf-8')

    # Three groups, committed in this order — see `_commit_in_stages` for why.
    ops: list = [
        CommitOperationAdd(
            path_in_repo  = DATA_ANN,
            path_or_fileobj = io.BytesIO(payload),
        )
    ]
    ops_crops: list = []
    ops_drain: list = []

    # 2. Copy any approved crop that isn't already in data/crops/.
    missing: list[str] = []
    new_crops = 0
    for sha in merged:
        dst = crop_path(sha)
        if dst in repo_files or f'{DATA_CRP}/{sha}.png' in repo_files:
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
        ops_crops.append(CommitOperationAdd(
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
                ops_drain.append(CommitOperationDelete(path_in_repo=staging_png))
                deleted_crops += 1

    # A row whose crop exists nowhere — not in staging, not in data/ — is the
    # mirror of the orphan swept below, and until now only that one direction
    # was covered. It can never be tallied, because tallying reads the PNG's
    # bytes, so it would sit in `staging/<iid>/annotations.jsonl` for good.
    # Two were there when this was written.
    #
    # Uploads write the rows and the PNGs in one commit, so a row can only
    # reach this state through a partial write or an out-of-band edit; either
    # way what is left is a reference to nothing.
    staged_shas = {Path(p).stem for p in repo_files
                   if p.startswith('staging/') and '/crops/' in p
                   and p.endswith('.png')}
    barred = set(rejected or ())
    surviving = _surviving_rows(staging_records, staged_shas, existing,
                                safe_promoted, barred)

    for iid, records in staging_records.items():
        kept = surviving[iid]
        dangling = len(records) - len(kept) - sum(
            1 for r in records
            if (r.get('crop_sha256') or '').strip() in safe_promoted)
        if dangling:
            print(f'Sweeping {dangling} staging row(s) for {iid[:8]} the '
                  f'tally can never reach.')
        if len(kept) == len(records):
            continue
        staging_ann = f'staging/{iid}/annotations.jsonl'
        if not kept:
            ops_drain.append(CommitOperationDelete(path_in_repo=staging_ann))
            deleted_annos += 1
        else:
            buf = ('\n'.join(json.dumps(r, ensure_ascii=False) for r in kept)
                   + '\n').encode('utf-8')
            ops_drain.append(CommitOperationAdd(
                path_in_repo    = staging_ann,
                path_or_fileobj = io.BytesIO(buf),
            ))
            rewritten_annos += 1

    # 4. Sweep staging crops no remaining row refers to.
    #
    # A crop is only ever tallied through its row in
    # `staging/<iid>/annotations.jsonl`, so a PNG with no row can never be
    # promoted, drained or seen again — it is dead weight that only a manual
    # script removed, and `admin_drain_stale_staging.py` had not run since
    # 2026-07-17. Ten such files were in staging when this was written.
    #
    # They are cheap to prevent rather than to remember: after the trim above,
    # anything under `staging/<iid>/crops/` whose sha is in no surviving row
    # is swept. Content-addressed and idempotent, and it cannot take a crop
    # that still has a vote — including a vote cast in this very batch, since
    # `safe_promoted` rows are the ones being drained anyway.
    kept_by_iid: dict[str, set[str]] = {}
    for iid, records in staging_records.items():
        kept_by_iid[iid] = {
            (r.get('crop_sha256') or '').strip() for r in records
            if (r.get('crop_sha256') or '').strip() not in safe_promoted
        }
    ann_iids = {p.split('/')[1] for p in repo_files
                if p.startswith('staging/') and p.endswith('/annotations.jsonl')}
    already_dropped = {op.path_in_repo for op in ops_drain}
    swept = 0
    for path in repo_files:
        if not (path.startswith('staging/') and '/crops/' in path
                and path.endswith('.png')):
            continue
        if path in already_dropped:
            continue
        parts = path.split('/')
        iid, sha = parts[1], Path(parts[-1]).stem
        if sha in barred:
            orphan = True
        elif iid in kept_by_iid:
            orphan = sha not in kept_by_iid[iid]
        else:
            # No annotations.jsonl for this install at all. Uploads write the
            # PNGs and the rows in one commit, so there is no window where a
            # crop legitimately has no row — an install in this state was
            # written by something that is not the upload path, and its crops
            # can never be tallied. `staging/migration-sister/` is the case
            # that occurred: a one-off migration left ten of them, and only a
            # manual script would ever have removed them.
            orphan = iid not in ann_iids
        if orphan:
            ops_drain.append(CommitOperationDelete(path_in_repo=path))
            swept += 1
    if swept:
        print(f'Sweeping {swept} staging crop(s) no annotation row refers to.')

    print(f'Committing: 1 annotations file + {new_crops} new crops + '
          f'drain({deleted_crops} stg crops, {swept} orphans swept, '
          f'{rewritten_annos} stg ann trimmed, {deleted_annos} stg ann '
          f'emptied)…')

    stamp = datetime.now(UTC).strftime('%Y-%m-%d %H:%M')
    _commit_in_stages(
        api,
        crop_ops   = ops_crops,
        anno_op    = ops[0],
        drain_ops  = ops_drain,
        summary    = (f'democratic_merge: {len(merged)} entries '
                      f'(+{new_crops} new crops, drained '
                      f'{deleted_crops} staging crops) @ {stamp} UTC'),
        chunk      = chunk,
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
    ap.add_argument('--chunk',   type=int, default=COMMIT_CHUNK, metavar='N',
                    help=f'Operations per commit (default {COMMIT_CHUNK}). '
                         f'One commit for everything is what broke; see '
                         f'COMMIT_CHUNK.')
    ap.add_argument('--min',     type=int, default=2, metavar='N',
                    help='Accepted for compatibility with the shared merge '
                         'workflow and NOT enforced here: every tallied entry '
                         'is applied and staging empties. Confidence is '
                         'recorded as the vote count, not used as a gate. '
                         'A value other than the default is reported at the '
                         'top of the run so it cannot look effective.')
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

    rejected = _load_rejected_shas(snap_dir)
    relabelled = _load_relabelled(snap_dir)
    if relabelled:
        print(f'Review ledger: {len(relabelled)} sha(s) pinned to a '
              f'maintainer-corrected name')
    if rejected:
        print(f'Review ledger: {len(rejected)} rejected sha(s) barred from data/')

    (name_votes, slot_votes, crop_src, per_install,
     contributors_for_sha, staging_records) = _collect_votes(
        snap_dir, since=args.since, repo_files=repo_files, rejected=rejected)
    print(f'Contributors: {len(per_install)}   '
          f'unique sha hashes voted on: {len(name_votes)}')

    # Not `if not name_votes: return` — the sweep exists for staging files
    # that no row refers to, which is exactly the state in which there are no
    # votes to tally. Returning here would make it unreachable in the only
    # situation it was written for.
    _ann_iids = {p.split('/')[1] for p in repo_files
                 if p.startswith('staging/') and p.endswith('/annotations.jsonl')}
    _voted = {(r.get('crop_sha256') or '').strip()
              for recs in staging_records.values() for r in recs}
    _staged_shas = {Path(p).stem for p in repo_files
                    if p.startswith('staging/') and '/crops/' in p
                    and p.endswith('.png')}
    # Three ways a staging file can be unreachable by the tally, and each
    # leaves it in place unless the sweep runs: no row refers to the PNG, the
    # install has no annotations file at all, or the sha is barred by the
    # review ledger. The mirror case — a row whose PNG exists nowhere — has
    # no PNG to find here, so it is looked for in the rows instead.
    _sweepable = any(
        p.startswith('staging/') and '/crops/' in p and p.endswith('.png')
        and (Path(p).stem not in _voted
             or p.split('/')[1] not in _ann_iids
             or Path(p).stem in rejected)
        for p in repo_files
    ) or any(
        (r.get('crop_sha256') or '').strip() not in _staged_shas
        and (r.get('crop_sha256') or '').strip() not in existing
        for recs in staging_records.values() for r in recs
    )
    # A correction overwritten before the pin existed needs no vote and no
    # staging file to restore, so it is invisible to both tests above. Third
    # time this early exit has hidden work from itself; check for it too.
    _unhealed = any(existing.get(sha, {}).get('name') not in (None, nm)
                    for sha, nm in relabelled.items() if sha in existing)
    if not name_votes and not _sweepable and not _unhealed:
        print('No staging entries to merge — nothing to do.')
        return 0
    if not name_votes and _unhealed:
        print('No votes to tally, but data/ holds a maintainer correction '
              'that was overwritten — running the merge to restore it.')
    if not name_votes:
        print('No votes to tally, but staging holds files no row refers to '
              '— running the sweep.')

    if args.min != 2:
        print(f'NOTE: --min {args.min} is not a gate for crops — every tallied '
              f'entry is applied and staging empties. The vote count is '
              f'recorded on the entry instead.')

    merged, report, promoted_shas = _merge(
        name_votes, slot_votes, existing,
        verbose=args.verbose, rejected=rejected, relabelled=relabelled)

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
           repo_files, contributors_for_sha, staging_records,
           rejected=rejected, chunk=args.chunk)
    print('OK — committed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
