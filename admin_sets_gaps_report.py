#!/usr/bin/env python3
"""Report which items SETS drops on import, across every install that saw one.

Clients upload their own ledger to `sets_gaps/<install_id>.json`; this reads
all of them and counts **how many distinct installs** hit each item. That is
the number an upstream request can be built on — "recognised in N separate
installs" is evidence, while "one user posted a screenshot" is not.

The two reasons are counted apart on purpose, because they are requests to
two different projects:

    missing-from-cargo   No cargo table stores the item, though the wiki
                         documents it. WARP reaches it through the harvested
                         overlay; SETS reads the same tables and cannot.
                         → ask the wiki for a cargo row.

    sets-loader-skips    Cargo stores the row and SETS's own build loader
                         passes over it (the Advanced/Elite hangars).
                         → ask SETS to accept it.

Uploads replace an install's file rather than adding to it, so an item that
stops being reported disappears from this report by itself. A name here is
therefore current, not historical.

Usage:
    python admin_sets_gaps_report.py
    python admin_sets_gaps_report.py --reason sets-loader-skips
    python admin_sets_gaps_report.py --min-installs 3
    python admin_sets_gaps_report.py --json
    python admin_sets_gaps_report.py --stale-after 0   # count every ledger

A ledger not refreshed in STALE_AFTER_DAYS is ignored: clients push daily, so
one that has stopped belongs to an install that is gone, and letting it keep
voting would inflate the only number this report exists to produce. Nothing is
deleted — an install that comes back is counted again on its next push.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from hf_clone import clone_hf_shallow

HF_ICONS_REPO_ID = os.environ.get('HF_ICONS_REPO_ID', 'sets-sto/sto-icon-dataset')
HF_TOKEN         = os.environ.get('HF_TOKEN', '')


def _load(root: Path) -> list[dict]:
    """Every per-install ledger currently published."""
    d = root / 'sets_gaps'
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob('*.json')):
        try:
            out.append(json.loads(p.read_text(encoding='utf-8')))
        except Exception as e:
            print(f'  skipped {p.name}: {e}', file=sys.stderr)
    return out


# A live install re-uploads daily, so a ledger that has not moved in this long
# belongs to one that is gone. Counting it would let installs that stopped
# playing keep voting forever, and the count is the entire argument. Nothing is
# deleted — the file simply stops being counted, so an install that comes back
# after a year is included again on its next push.
STALE_AFTER_DAYS = 90


def _is_stale(ledger: dict, now: datetime, max_age_days: int) -> bool:
    """True if this install has not refreshed its ledger recently enough.

    A ledger with no or unparseable `updated_at` predates the field and is
    treated as stale: it cannot be shown to be current, and over-counting is
    the failure that matters here.
    """
    raw = (ledger.get('updated_at') or '').strip()
    if not raw:
        return True
    try:
        ts = datetime.fromisoformat(raw.replace('Z', '+00:00'))
    except ValueError:
        return True
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (now - ts).days > max_age_days


def _tally(ledgers: list[dict]) -> dict[tuple[str, str], dict]:
    """Collapse per-install ledgers into one entry per (reason, name).

    An install is counted once per item however many times it reported it —
    the file is a snapshot, not a log, so there is nothing to add up.
    """
    agg: dict[tuple[str, str], dict] = defaultdict(
        lambda: {'installs': 0, 'slots': set()})
    for led in ledgers:
        seen: set[tuple[str, str]] = set()
        for item in led.get('items') or []:
            name   = (item.get('name') or '').strip()
            reason = (item.get('reason') or '').strip()
            if not name or not reason:
                continue
            key = (reason, name)
            if key in seen:
                continue          # one install, one vote
            seen.add(key)
            agg[key]['installs'] += 1
            agg[key]['slots'].update(item.get('slots') or [])
    return agg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reason', default='', help='only this reason')
    ap.add_argument('--min-installs', type=int, default=1,
                    help='hide items seen by fewer installs than this')
    ap.add_argument('--json', action='store_true', help='machine-readable output')
    ap.add_argument('--stale-after', type=int, default=STALE_AFTER_DAYS,
                    metavar='DAYS',
                    help=f'ignore ledgers not refreshed in this many days '
                         f'(default {STALE_AFTER_DAYS}); 0 disables')
    args = ap.parse_args()

    if not HF_TOKEN:
        print('ERROR: HF_TOKEN is not set.', file=sys.stderr)
        return 2
    root = clone_hf_shallow(HF_ICONS_REPO_ID, HF_TOKEN, repo_type='dataset')
    all_ledgers = _load(root)
    if args.stale_after > 0:
        now = datetime.now(timezone.utc)
        ledgers = [l for l in all_ledgers
                   if not _is_stale(l, now, args.stale_after)]
    else:
        ledgers = all_ledgers
    stale = len(all_ledgers) - len(ledgers)
    agg = _tally(ledgers)

    rows = [
        {'reason': r, 'name': n,
         'installs': v['installs'], 'slots': sorted(v['slots'])}
        for (r, n), v in agg.items()
        if v['installs'] >= args.min_installs
        and (not args.reason or r == args.reason)
    ]
    rows.sort(key=lambda e: (e['reason'], -e['installs'], e['name']))

    if args.json:
        print(json.dumps({'installs_reporting': len(ledgers),
                          'installs_stale': stale, 'items': rows},
                         ensure_ascii=False, indent=2))
        return 0

    print(f'Ledgers from {len(ledgers)} install(s)'
          + (f'  ({stale} ignored as stale — no refresh in '
             f'{args.stale_after} days)' if stale else '') + '\n')
    if not rows:
        print('No items reported.')
        return 0

    where = {
        'missing-from-cargo': 'ask the wiki — no cargo table stores these',
        'sets-loader-skips':  "ask SETS — its loader passes over these",
    }
    current = ''
    for row in rows:
        if row['reason'] != current:
            current = row['reason']
            print(f'\n{current}  ({where.get(current, "")})')
            print(f'  {"installs":>8}  item')
        slots = ', '.join(row['slots'])
        print(f'  {row["installs"]:>8}  {row["name"]}'
              + (f'   [{slots}]' if slots else ''))
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
