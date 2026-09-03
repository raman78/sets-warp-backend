#!/usr/bin/env python3
"""
admin_merge.py — WARP Knowledge Base merger
============================================
Reads all contributions from HF Dataset, performs majority-vote,
saves the result to knowledge.json.

Usage:
    python admin_merge.py                    # dry-run (report only)
    python admin_merge.py --apply            # save knowledge.json to HF
    python admin_merge.py --apply --min 1    # 1 vote is enough (default is 2)
    python admin_merge.py --since 2026-03-01 # only contributions from this date onwards

Environment variables (or .env file):
    HF_TOKEN     — HF write token
    HF_REPO_ID   — e.g., sets-sto/warp-knowledge
    ADMIN_KEY    — admin key (optional, for hitting the /admin/merge endpoint)
"""

from __future__ import annotations

from hf_commit import commit_adds_then_deletes

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, date, timezone
UTC = timezone.utc
from pathlib import Path

# ── Auto-restart in .venv if needed ───────────────────────────────────────

def _ensure_venv():
    """
    Fully standalone — zero system Python, zero system pip.

    1. If already in local .venv → OK
    2. If .venv exists → restart in it
    3. If .venv is missing → run setup.py which:
         - downloads portable Python 3.12 to .python/
         - creates .venv from that Python
         - installs requirements.txt (including huggingface-hub)
       Then restart in the ready .venv

    CI environments (GitHub Actions etc.) skip this — they install
    dependencies via workflow steps and run with system Python.
    """
    if os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS') or os.environ.get('WARP_NO_VENV'):
        return

    here     = Path(__file__).resolve().parent
    is_win   = sys.platform == 'win32'
    venv_py  = here / ('.venv/Scripts/python.exe' if is_win else '.venv/bin/python')
    setup_py = here / 'setup.py'

    # 1. Already in our .venv
    if venv_py.exists() and Path(sys.executable).resolve() == venv_py.resolve():
        return

    # 2. .venv exists — restart in it
    if venv_py.exists():
        os.execv(str(venv_py), [str(venv_py)] + sys.argv)

    # 3. Missing .venv — run setup.py (downloads portable Python, builds venv)
    if setup_py.exists():
        print('  → Missing .venv — running setup.py (portable Python 3.12) ...')
        # setup.py is interactive — we run it and upon completion
        # restart in the newly created .venv
        ret = subprocess.call([sys.executable, str(setup_py)])
        if ret != 0:
            print('ERROR: setup.py failed.', file=sys.stderr)
            sys.exit(1)
        if venv_py.exists():
            os.execv(str(venv_py), [str(venv_py)] + sys.argv)
        else:
            print('ERROR: setup.py did not create .venv.', file=sys.stderr)
            sys.exit(1)
    else:
        print('ERROR: setup.py missing — run it manually to configure the environment.',
              file=sys.stderr)
        sys.exit(1)

_ensure_venv()


# ── Load .env if present ───────────────────────────────────────────────────────

def _load_env():
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()

HF_TOKEN   = os.environ.get('HF_TOKEN', '')
HF_REPO_ID = os.environ.get('HF_REPO_ID', 'sets-sto/warp-knowledge')


# ── HF helpers ─────────────────────────────────────────────────────────────────

def _hf_list_contributions(
    processed_ids:  set[str] | None = None,
    watermark_date: str | None      = None,
    since:          str | None      = None,
) -> tuple[list[dict], list[str], list[Path]]:
    """
    Fetches contribution JSONs from HF Dataset.

    Uses a shallow git clone of the HF dataset repo (see hf_clone.py) —
    one network round-trip instead of per-file HEAD requests, which avoids
    HF's HTTP 429 limiter as the tree grows.

    Returns (contribs, new_ids, all_paths) where:
      - contribs: parsed JSON records for contributions NOT in processed_ids
        and whose date directory is >= watermark_date.
      - new_ids: contribution IDs (file stems) corresponding to those records,
        in the same order.
      - all_paths: every contribution file path on disk (regardless of
        filters). Used by compaction to map id → date.
    """
    if not HF_TOKEN or not HF_REPO_ID:
        print('ERROR: HF_TOKEN or HF_REPO_ID not set', file=sys.stderr)
        sys.exit(1)

    processed_ids = processed_ids or set()

    print(f'Cloning contributions tree (shallow)...')
    from hf_clone import clone_hf_shallow
    snap_dir = clone_hf_shallow(HF_REPO_ID, HF_TOKEN, repo_type='dataset')

    contrib_root = Path(snap_dir) / 'contributions'
    if not contrib_root.exists():
        print(f'WARNING: no contributions/ folder found at {contrib_root}')
        return [], [], []

    all_paths = sorted(contrib_root.glob('*/*.json'))
    total_on_disk = len(all_paths)

    # Apply watermark + since filters (lexicographic on YYYY-MM-DD date dir).
    effective_since = max(s for s in (watermark_date, since) if s) if (watermark_date or since) else None
    json_paths = all_paths
    if effective_since:
        json_paths = [p for p in json_paths if p.parent.name >= effective_since]

    # Drop already-processed IDs (file stem == contribution_id).
    json_paths = [p for p in json_paths if p.stem not in processed_ids]

    print(f'Found {total_on_disk} contribution files on HF; '
          f'{len(json_paths)} new to process'
          + (f' (since {effective_since})' if effective_since else '')
          + (f', skipping {len(processed_ids)} already processed' if processed_ids else ''))

    contribs: list[dict] = []
    new_ids:  list[str]  = []
    for p in json_paths:
        try:
            contribs.append(json.loads(p.read_text(encoding='utf-8')))
            new_ids.append(p.stem)
        except Exception as e:
            print(f'  SKIP {p.name}: {e}')

    return contribs, new_ids, all_paths


def _hf_load_state() -> tuple[dict[str, str], set[str], str]:
    """
    Loads the current knowledge.json from HF.

    Returns (knowledge, processed_contribution_ids, watermark_date).
    Backwards-compatible with old knowledge.json files that lack the
    processed_contributions / watermark_date fields — both default to empty.
    """
    try:
        from huggingface_hub import hf_hub_download
        from hf_retry import retry_on_429
        # Wrap the download so a transient HF 429 doesn't fall into the
        # "starting from scratch" branch and overwrite a good knowledge.json.
        local = retry_on_429(
            lambda: hf_hub_download(
                HF_REPO_ID, 'knowledge.json',
                repo_type='dataset', token=HF_TOKEN or None,
            ),
            label='hf_hub_download(knowledge.json)',
        )
        data = json.loads(Path(local).read_text(encoding='utf-8'))
        knowledge = data.get('knowledge', data) if isinstance(data, dict) else {}
        if not isinstance(knowledge, dict):
            knowledge = {}
        processed = set(data.get('processed_contributions', [])) if isinstance(data, dict) else set()
        watermark = data.get('watermark_date', '') if isinstance(data, dict) else ''
        return knowledge, processed, watermark
    except Exception as e:
        print(f'NOTICE: knowledge.json does not exist or error occurred ({e}) — starting from scratch')
        return {}, set(), ''


# Compaction threshold: when processed_contributions grows past this, advance
# watermark_date to drop the older half (still implicitly considered processed
# because their date < watermark_date).
_PROCESSED_COMPACTION_THRESHOLD = 5000


def _compact_processed(
    processed_ids:  set[str],
    watermark_date: str,
    seen_paths:     list[Path],
) -> tuple[list[str], str]:
    """
    Advance watermark_date if processed list grows too large.

    Strategy: when list exceeds threshold, sort all known contribution files
    by date, pick the date at the midpoint as new watermark, drop IDs whose
    date < new watermark. They're implicitly considered processed because
    the watermark gates all future runs.
    """
    if len(processed_ids) <= _PROCESSED_COMPACTION_THRESHOLD:
        return sorted(processed_ids), watermark_date

    # Build id → date_dir map from the snapshot we just read.
    id_to_date = {p.stem: p.parent.name for p in seen_paths}
    by_date = sorted(
        ((id_to_date.get(i, ''), i) for i in processed_ids),
        key=lambda t: t[0],
    )
    midpoint = len(by_date) // 2
    new_watermark = by_date[midpoint][0]
    if not new_watermark or new_watermark <= watermark_date:
        # Can't advance — return as-is.
        return sorted(processed_ids), watermark_date

    kept = [i for d, i in by_date if d >= new_watermark]
    dropped = len(processed_ids) - len(kept)
    print(f'[compact] watermark_date {watermark_date or "(none)"} → {new_watermark}; '
          f'dropped {dropped} ids implicitly covered by watermark')
    return sorted(kept), new_watermark


def _hf_save_state(
    knowledge:        dict[str, str],
    processed_ids:    list[str],
    watermark_date:   str,
    losers_by_phash:  dict[str, dict[str, int]] | None = None,
    drain_contribs:   list[Path] | None              = None,
) -> bool:
    """Save knowledge.json + (optionally) drain promoted contributions in a
    single atomic HF commit.

    D-B.5: `losers_by_phash` is persisted as a top-level field so dissent
    survives in the artefact (the runtime `knowledge` map stays a plain
    `phash → name` to preserve the client API contract).

    D-G.9: `drain_contribs` is a list of paths on the local snapshot pointing
    at contribution files whose phash made consensus. Both `<uuid>.json` and
    `<uuid>.png` get a CommitOperationDelete in the same commit as the
    knowledge.json write — no half-applied state.
    """
    try:
        from huggingface_hub import (
            HfApi, CommitOperationAdd, CommitOperationDelete,
        )
        import io as _io
        api = HfApi(token=HF_TOKEN)
        payload_obj = {
            'schema_version':           2,
            'knowledge':                knowledge,
            'updated_at':               datetime.now(UTC).isoformat() + 'Z',
            'entries':                  len(knowledge),
            'processed_contributions':  processed_ids,
            'watermark_date':           watermark_date,
        }
        if losers_by_phash:
            payload_obj['losers'] = losers_by_phash
        payload = json.dumps(payload_obj, ensure_ascii=False, indent=2).encode('utf-8')

        ops: list = [CommitOperationAdd(
            path_in_repo  ='knowledge.json',
            path_or_fileobj=_io.BytesIO(payload),
        )]

        deleted = 0
        for p in (drain_contribs or []):
            # Each contribution file lives under contributions/YYYY-MM-DD/.
            # The snapshot root is two parents up (snapshot/contributions/<date>/uuid.json).
            rel = f'contributions/{p.parent.name}/{p.name}'
            ops.append(CommitOperationDelete(path_in_repo=rel))
            deleted += 1
            png = p.with_suffix('.png')
            rel_png = f'contributions/{p.parent.name}/{png.name}'
            ops.append(CommitOperationDelete(path_in_repo=rel_png))
            deleted += 1

        # Chunked, additions before deletions — see hf_commit for why one
        # commit for everything is no longer safe at these volumes.
        commit_adds_then_deletes(
            api, HF_REPO_ID, 'dataset', ops,
            (f'admin_merge: {len(knowledge)} entries, '
             f'{len(processed_ids)} tracked, '
             f'drained {deleted // 2} promoted contribs '
             f'({datetime.now(UTC).strftime("%Y-%m-%d %H:%M")} UTC)'),
        )
        return True
    except Exception as e:
        print(f'ERROR: saving knowledge.json: {e}', file=sys.stderr)
        return False


# ── Merge logic ────────────────────────────────────────────────────────────────

def _is_poison_name(name: str) -> bool:
    """
    Names that must NEVER enter knowledge.json:
      - virtual classes (__empty__, __inactive__, __boff_*) — Stage 0 would
        hard-override real icons to "empty/inactive" with conf=1.0.
      - leftover dev-test entries.
    """
    return name.startswith('__') or name == 'Test Item Name'


def merge(
    contribs:   list[dict],
    existing:   dict[str, str],
    min_votes:  int = 2,
    verbose:    bool = False,
) -> tuple[dict[str, str], list[dict], dict[str, dict[str, int]], dict[str, set[str]]]:
    """
    Majority-vote merge.

    Returns (merged_knowledge, report_rows, losers_by_phash, contribs_by_phash):
      - merged_knowledge[phash]   → winning name (string — runtime API contract).
      - report_rows               → display dicts (for printing + summary stats).
      - losers_by_phash[phash]    → minority {name: count} (top 3), populated
        only for entries that actually made it into merged_knowledge. Stored
        as a parallel top-level field in knowledge.json (D-B.5).
      - contribs_by_phash[phash]  → set of contribution_id whose vote landed
        on that phash. The drain (D-G.9) uses this to issue
        CommitOperationDelete for every uuid that contributed to a promoted
        phash, draining contributions/ after consensus.
    """
    # Group by phash
    phash_votes: dict[str, Counter] = {}
    phash_meta:  dict[str, dict]    = {}   # phash → {total, confirmed, wrong_names}
    contribs_by_phash: dict[str, set[str]] = {}

    for c in contribs:
        if not isinstance(c, dict):
            continue
        ph   = c.get('phash', '').strip()
        name = c.get('item_name', '').strip()
        if not ph or not name:
            continue
        if _is_poison_name(name):
            continue

        phash_votes.setdefault(ph, Counter())[name] += 1
        meta = phash_meta.setdefault(ph, {'total': 0, 'confirmed': 0, 'wrong': Counter()})
        meta['total'] += 1
        if c.get('confirmed'):
            meta['confirmed'] += 1
        wrong = c.get('wrong_name', '').strip()
        if wrong:
            meta['wrong'][wrong] += 1
        cid = (c.get('contribution_id') or '').strip()
        if cid:
            contribs_by_phash.setdefault(ph, set()).add(cid)

    # Drop already-merged poison entries (legacy data from before the filter
    # existed). This rewrites knowledge.json to a clean state on next merge.
    merged  = {ph: nm for ph, nm in existing.items() if not _is_poison_name(nm)}
    report  = []
    losers_by_phash: dict[str, dict[str, int]] = {}
    _dropped_poison = len(existing) - len(merged)
    if _dropped_poison > 0:
        print(f'[clean] dropped {_dropped_poison} legacy poison entries '
              f'(virtual classes / test rows)')

    for ph, votes in sorted(phash_votes.items()):
        winner, count = votes.most_common(1)[0]
        meta          = phash_meta[ph]
        old_name      = existing.get(ph, '')

        # Threshold: min_votes, unless the phash is new — then 1 is enough
        threshold = min_votes if ph in existing else 1
        accepted  = count >= threshold

        action = 'SKIP'
        if accepted:
            if old_name == winner:
                action = 'unchanged'
            elif old_name:
                action = 'UPDATE'
                merged[ph] = winner
            else:
                action = 'NEW'
                merged[ph] = winner
            losers_top = {n: v for n, v in votes.most_common()[1:4] if n != winner}
            if losers_top:
                losers_by_phash[ph] = losers_top

        row = {
            'phash':    ph,
            'winner':   winner,
            'votes':    count,
            'total':    meta['total'],
            'old_name': old_name,
            'action':   action,
        }
        if meta['wrong']:
            row['wrong'] = dict(meta['wrong'].most_common(3))
        report.append(row)

        if verbose or action in ('NEW', 'UPDATE', 'SKIP'):
            _print_row(row)

    return merged, report, losers_by_phash, contribs_by_phash


def _print_row(row: dict):
    action  = row['action']
    symbol  = {'NEW': '✓', 'UPDATE': '↺', 'unchanged': '·', 'SKIP': '✗'}.get(action, '?')
    color   = {'NEW': '\033[92m', 'UPDATE': '\033[93m', 'SKIP': '\033[91m', 'unchanged': ''}.get(action, '')
    reset   = '\033[0m' if color else ''
    ph      = row['phash']
    winner  = row['winner'][:50]
    votes   = row['votes']
    total   = row['total']
    old     = f" (was: {row['old_name'][:30]})" if row.get('old_name') and action == 'UPDATE' else ''
    wrong   = f" [wrong: {list(row.get('wrong', {}).keys())}]" if row.get('wrong') else ''
    print(f'  {color}{symbol} [{ph}] {winner!r:50s} {votes}/{total} votes{old}{wrong}{reset}')


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='WARP Knowledge Base merger — merges contributions from HF into knowledge.json',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python admin_merge.py                           # preview (dry-run)
  python admin_merge.py --apply                   # save to HF
  python admin_merge.py --apply --min 1           # 1 vote is enough
  python admin_merge.py --since 2026-03-01        # only from this date
  python admin_merge.py --apply --export k.json   # save locally + HF
  python admin_merge.py --verbose                 # show all entries

Environment variables (.env):
  HF_TOKEN    — HF write token (required)
  HF_REPO_ID  — e.g., sets-sto/warp-knowledge (default)
""",
    )
    parser.add_argument('--apply',   action='store_true',
                        help='Save result to HF (default: dry-run)')
    parser.add_argument('--min',     type=int, default=2, metavar='N',
                        help='Minimum number of votes (default: 2)')
    parser.add_argument('--since',   metavar='YYYY-MM-DD',
                        help='Include contributions only from this date onwards')
    parser.add_argument('--verbose', action='store_true',
                        help='Show all entries (not just changes)')
    parser.add_argument('--export',  metavar='FILE',
                        help='Save resulting knowledge.json locally')
    args = parser.parse_args()

    print('=' * 60)
    print(f'WARP Knowledge Merger')
    print(f'Repo:     {HF_REPO_ID}')
    print(f'Min votes: {args.min}')
    print(f'Mode:     {"APPLY" if args.apply else "DRY-RUN"}')
    if args.since:
        print(f'Since:    {args.since}')
    print('=' * 60)

    # 1. Load current state (knowledge + processed contribution IDs + watermark)
    existing, processed_ids, watermark = _hf_load_state()
    print(f'Current knowledge.json: {len(existing)} entries, '
          f'{len(processed_ids)} tracked contribution IDs, '
          f'watermark_date={watermark or "(none)"}\n')

    # 2. List contributions, filtered to NEW ones only.
    contribs, new_ids, all_paths = _hf_list_contributions(
        processed_ids  = processed_ids,
        watermark_date = watermark,
        since          = args.since,
    )
    if not contribs:
        print('No new contributions to process — nothing to do.')
        # Still rewrite knowledge.json in --apply mode to refresh updated_at?
        # No — would create empty commits on every cron run. Just exit.
        return

    confirmed = sum(1 for c in contribs if c.get('confirmed'))
    print(f'\nLoaded {len(contribs)} new contributions ({confirmed} confirmed)\n')

    # 3. Merge
    merged, report, losers_by_phash, contribs_by_phash = merge(
        contribs, existing, min_votes=args.min, verbose=args.verbose,
    )

    # 4. Report
    new_count     = sum(1 for r in report if r['action'] == 'NEW')
    update_count  = sum(1 for r in report if r['action'] == 'UPDATE')
    skip_count    = sum(1 for r in report if r['action'] == 'SKIP')
    unchanged     = sum(1 for r in report if r['action'] == 'unchanged')

    print(f'\n--- Summary ---')
    print(f'  ✓ New:      {new_count}')
    print(f'  ↺ Updated:  {update_count}')
    print(f'  · Unchanged: {unchanged}')
    print(f'  ✗ Skipped (not enough votes): {skip_count}')
    print(f'  Total after merge: {len(merged)} entries')

    # Uniform drain monitor (post-audit TODO #5). The contributions drain set
    # is built at apply-time below; here we report the upper bound — the
    # promoted phash count is a per-cycle proxy for drain pressure.
    promoted_for_log = sum(1 for r in report
                           if r['action'] in ('NEW', 'UPDATE', 'unchanged'))
    print(f'DRAIN: domain=contributions promoted={promoted_for_log} '
          f'new={new_count} update={update_count} skip={skip_count}')

    # 5. Local export
    if args.export:
        Path(args.export).write_text(
            json.dumps({'knowledge': merged, 'updated_at': datetime.now(UTC).isoformat() + 'Z'},
                       ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        print(f'\nSaved locally: {args.export}')

    # 6. Apply — update knowledge + processed_ids + watermark atomically (one commit)
    if args.apply:
        # Always advance processed_ids in --apply mode, even when the merge
        # added no new knowledge entries. Otherwise the same contributions
        # would be reprocessed every run (low-vote SKIPs would be re-evaluated
        # every time, which is fine but wastes downloads).
        updated_processed = processed_ids | set(new_ids)
        compacted_ids, new_watermark = _compact_processed(
            updated_processed, watermark, all_paths,
        )

        # D-G.9: identify which on-disk contribution files belong to a
        # promoted phash so we can delete them in the same commit. SKIP
        # contributions are kept — they may accumulate more votes later.
        promoted_phashes = {r['phash'] for r in report
                            if r['action'] in ('NEW', 'UPDATE', 'unchanged')}
        promoted_cids: set[str] = set()
        for ph in promoted_phashes:
            promoted_cids |= contribs_by_phash.get(ph, set())
        drain_paths = [p for p in all_paths if p.stem in promoted_cids]

        if new_count == 0 and update_count == 0 and new_watermark == watermark and not drain_paths:
            # Same knowledge, same watermark — still worth writing back to
            # advance processed_ids so we don't re-pull the same SKIPs.
            print(f'\nNo knowledge changes, but advancing tracked IDs '
                  f'({len(processed_ids)} → {len(compacted_ids)})...')
        else:
            print(f'\nSaving {len(merged)} entries to HF '
                  f'(tracked IDs: {len(compacted_ids)}, watermark: {new_watermark or "(none)"}, '
                  f'draining {len(drain_paths)} promoted contributions)...')

        ok = _hf_save_state(merged, compacted_ids, new_watermark,
                            losers_by_phash=losers_by_phash,
                            drain_contribs=drain_paths)
        if ok:
            print('OK — knowledge.json updated on HF.')
        else:
            print('ERROR — save failed.', file=sys.stderr)
            sys.exit(1)
    else:
        print('\nDRY-RUN — use --apply to save.')


if __name__ == '__main__':
    main()
