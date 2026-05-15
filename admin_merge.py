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
    """
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

def _hf_list_contributions(since: str | None = None) -> list[dict]:
    """
    Fetches all contribution JSONs from HF Dataset.

    Uses snapshot_download with allow_patterns=['contributions/**/*.json'] so
    that the entire contributions tree is fetched in parallel (8 workers by
    default) in a single call. PNGs are excluded by the pattern. Subsequent
    runs reuse the local HF cache, so only new contributions are downloaded
    on each invocation.
    """
    if not HF_TOKEN or not HF_REPO_ID:
        print('ERROR: HF_TOKEN or HF_REPO_ID not set', file=sys.stderr)
        sys.exit(1)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print('ERROR: pip install huggingface-hub', file=sys.stderr)
        sys.exit(1)

    # Build allow_patterns. If `since` is given, restrict to date subfolders
    # whose name is >= since (lexicographic comparison works on ISO YYYY-MM-DD).
    # We can't filter list_repo_tree-style here, so we just rely on the file
    # path containing the date and post-filter after download. The download
    # still benefits from parallelism even if we slightly over-fetch.
    print(f'Downloading contributions tree (parallel)...')
    snap_dir = snapshot_download(
        repo_id        = HF_REPO_ID,
        repo_type      = 'dataset',
        token          = HF_TOKEN,
        allow_patterns = ['contributions/**/*.json'],
        max_workers    = 16,
    )

    contrib_root = Path(snap_dir) / 'contributions'
    if not contrib_root.exists():
        print(f'WARNING: no contributions/ folder found at {contrib_root}')
        return []

    json_paths = sorted(contrib_root.glob('*/*.json'))
    if since:
        json_paths = [p for p in json_paths if p.parent.name >= since]

    print(f'Found {len(json_paths)} contribution files'
          + (f' since {since}' if since else ''))

    contribs = []
    for p in json_paths:
        try:
            contribs.append(json.loads(p.read_text(encoding='utf-8')))
        except Exception as e:
            print(f'  SKIP {p.name}: {e}')

    return contribs


def _hf_load_knowledge() -> dict[str, str]:
    """Loads the current knowledge.json from HF."""
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            HF_REPO_ID, 'knowledge.json',
            repo_type='dataset', token=HF_TOKEN or None,
        )
        data = json.loads(Path(local).read_text(encoding='utf-8'))
        return data.get('knowledge', data)
    except Exception as e:
        print(f'NOTICE: knowledge.json does not exist or error occurred ({e}) — starting from scratch')
        return {}


def _hf_save_knowledge(knowledge: dict[str, str]) -> bool:
    """Saves knowledge.json to HF Dataset."""
    try:
        from huggingface_hub import HfApi
        api     = HfApi(token=HF_TOKEN)
        payload = json.dumps(
            {
                'knowledge':  knowledge,
                'updated_at': datetime.now(UTC).isoformat() + 'Z',
                'entries':    len(knowledge),
            },
            ensure_ascii=False,
            indent=2,
        ).encode('utf-8')
        api.upload_file(
            path_or_fileobj=payload,
            path_in_repo='knowledge.json',
            repo_id=HF_REPO_ID,
            repo_type='dataset',
            commit_message=f'admin_merge: {len(knowledge)} entries '
                           f'({datetime.now(UTC).strftime("%Y-%m-%d %H:%M")} UTC)',
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
) -> tuple[dict[str, str], list[dict]]:
    """
    Majority-vote merge.

    Returns (merged_knowledge, report_rows).
    report_rows — list of dictionaries for display.
    """
    # Group by phash
    phash_votes: dict[str, Counter] = {}
    phash_meta:  dict[str, dict]    = {}   # phash → {total, confirmed, wrong_names}

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

    # Drop already-merged poison entries (legacy data from before the filter
    # existed). This rewrites knowledge.json to a clean state on next merge.
    merged  = {ph: nm for ph, nm in existing.items() if not _is_poison_name(nm)}
    report  = []
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

    return merged, report


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

    # 1. Load contributions
    contribs = _hf_list_contributions(since=args.since)
    if not contribs:
        print('No contributions found — nothing to do.')
        return

    confirmed = sum(1 for c in contribs if c.get('confirmed'))
    print(f'\nLoaded {len(contribs)} contributions ({confirmed} confirmed)\n')

    # 2. Load current knowledge
    existing = _hf_load_knowledge()
    print(f'Current knowledge.json: {len(existing)} entries\n')

    # 3. Merge
    merged, report = merge(contribs, existing, min_votes=args.min, verbose=args.verbose)

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

    # 5. Local export
    if args.export:
        Path(args.export).write_text(
            json.dumps({'knowledge': merged, 'updated_at': datetime.now(UTC).isoformat() + 'Z'},
                       ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        print(f'\nSaved locally: {args.export}')

    # 6. Apply
    if args.apply:
        if new_count == 0 and update_count == 0:
            print('\nNo changes — knowledge.json will not be overwritten.')
            return
        print(f'\nSaving {len(merged)} entries to HF...')
        ok = _hf_save_knowledge(merged)
        if ok:
            print('OK — knowledge.json updated on HF.')
        else:
            print('ERROR — save failed.', file=sys.stderr)
            sys.exit(1)
    else:
        print('\nDRY-RUN — use --apply to save.')


if __name__ == '__main__':
    main()
