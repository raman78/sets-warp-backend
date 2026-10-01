#!/usr/bin/env python3
"""
admin_rebuild_votes.py — rebuild knowledge.json `voters` from history
======================================================================
One-shot migration of knowledge.json to schema 4.

Schema 3 kept `votes`, phash → {name: count}, where the count was of
contribution files. One install voting the same name for one hash 27
times counted 27, and votes processed before the tally existed (2026-09-25)
were not counted at all: 205 dissents and about 2,950 agreeing votes sat
on disk with no effect. Schema 4 keeps `voters`, phash → {install_id:
[name, timestamp]}, one vote per install, and admin_merge refuses to run
until it exists.

What this does:
    1. Full-history clone of the knowledge repo (no LFS blobs).
    2. Read every contributions/<date>/<id>.json ever added — including the
       ones drained since, which survive in history.
    3. Leave out votes for a (phash, name) pair that admin_scrub_knowledge
       removed, if cast before that scrub. A scrub is read from the
       knowledge.json diff of each `admin_scrub_knowledge:` commit.
    4. Feed them all to admin_merge.merge() with empty voters and today's
       knowledge map as the incumbent — the same rule as every merge run.
    5. Report every entry that would change. --apply writes knowledge.json
       (schema 4) in one commit. Nothing is deleted here; the backlog of
       counted files is drained afterwards by drain_stale_staging.yml.

Usage:
    .venv/bin/python admin_rebuild_votes.py            # dry-run report
    .venv/bin/python admin_rebuild_votes.py --apply    # write schema 4
    .venv/bin/python admin_rebuild_votes.py --min 3    # same flag as admin_merge

Environment variables (.env, same as admin_merge.py):
    HF_TOKEN     — HF write token
    HF_REPO_ID   — default: sets-sto/warp-knowledge
"""

from __future__ import annotations

import admin_merge   # .venv + .env bootstrap happens on import

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from hf_clone import clone_hf_shallow


# ── History ───────────────────────────────────────────────────────────────────

def _git(repo: Path, *args: str, stdin: bytes | None = None) -> bytes:
    return subprocess.run(['git', *args], cwd=repo, input=stdin,
                          capture_output=True, check=True).stdout


def _cat_files(repo: Path, specs: list[str]) -> list[bytes]:
    """Blob contents for `<rev>:<path>` specs, in order, in one git call."""
    out = _git(repo, 'cat-file', '--batch',
               stdin=('\n'.join(specs) + '\n').encode())
    blobs: list[bytes] = []
    i = 0
    while i < len(out):
        nl     = out.index(b'\n', i)
        header = out[i:nl].split()
        if header[-1] == b'missing':
            raise RuntimeError(f'git cat-file: {header[0].decode()} missing')
        size = int(header[2])
        blobs.append(out[nl + 1:nl + 1 + size])
        i = nl + 1 + size + 1
    return blobs


def _all_contributions(repo: Path) -> list[dict]:
    """Every contribution JSON ever added to the repo, deleted or not."""
    log = _git(repo, 'log', '--diff-filter=A', '--format=C %H', '--name-only',
               '--', 'contributions/*.json').decode()
    specs: list[str] = []
    commit = ''
    for line in log.splitlines():
        if line.startswith('C '):
            commit = line[2:]
        elif line.strip():
            specs.append(f'{commit}:{line.strip()}')
    contribs: list[dict] = []
    for spec, blob in zip(specs, _cat_files(repo, specs)):
        try:
            contribs.append(json.loads(blob))
        except ValueError as e:
            print(f'  SKIP unreadable {spec.split(":", 1)[1]}: {e}')
    return contribs


def _scrubbed_pairs(repo: Path) -> dict[tuple[str, str], str]:
    """(phash, name) → ISO time of the admin_scrub_knowledge commit that removed it."""
    log = _git(repo, 'log', '--format=%H %cI %s', '--', 'knowledge.json').decode()
    scrubs = [line.split(' ', 2) for line in log.splitlines()
              if line.split(' ', 2)[2].startswith('admin_scrub_knowledge')]
    pairs: dict[tuple[str, str], str] = {}
    for sha, when, _ in scrubs:
        before, after = (json.loads(b) for b in _cat_files(
            repo, [f'{sha}^:knowledge.json', f'{sha}:knowledge.json']))
        kb = before.get('knowledge', before)
        ka = after.get('knowledge', after)
        for ph, name in kb.items():
            if ka.get(ph) != name:
                pairs[(ph, name)] = when
    return pairs


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--apply', action='store_true',
                    help='Write knowledge.json (schema 4) to HF (default: dry-run)')
    ap.add_argument('--min', type=int, default=2, metavar='N',
                    help='Installs a challenger needs to change an entry (default: 2)')
    args = ap.parse_args()

    existing, processed, watermark, voters_now = admin_merge._hf_load_state()
    print(f'knowledge.json: {len(existing)} entries, '
          f'{"schema 4 (voters present — rebuilding anyway)" if voters_now is not None else "no voters"}')

    print('Cloning full history of the knowledge repo...')
    repo = clone_hf_shallow(admin_merge.HF_REPO_ID, admin_merge.HF_TOKEN,
                            repo_type='dataset', full_history=True)
    contribs = _all_contributions(repo)
    print(f'  {len(contribs)} contributions in history, '
          f'from {len({c.get("install_id") for c in contribs})} installs')

    scrubbed = _scrubbed_pairs(repo)
    kept, dropped = [], 0
    for c in contribs:
        key  = ((c.get('phash') or '').strip(), (c.get('item_name') or '').strip())
        when = scrubbed.get(key)
        if when and str(c.get('timestamp') or '') < when:
            dropped += 1
            continue
        kept.append(c)
    print(f'  {len(scrubbed)} (phash, name) pairs removed by a scrub; '
          f'{dropped} votes for them, cast before the scrub, left out\n')

    merged, report, voters, _ = admin_merge.merge(
        kept, existing, min_votes=args.min, voters={})

    actions = Counter(r['action'] for r in report)
    changed = [r for r in report if r['action'] in ('NEW', 'UPDATE')]
    no_voter = [ph for ph in merged if not voters.get(ph)]
    pairs = sum(len(by) for by in voters.values())
    print(f'\n--- Rebuild ---')
    print(f'  {pairs} install votes on {len(voters)} hashes '
          f'(from {len(kept)} contribution files)')
    print(f'  Actions: {dict(actions)}')
    print(f'  Entries with no recorded voter (count one vote for their name): {len(no_voter)}')
    print(f'  Entries that change: {len(changed)}')
    tally = admin_merge.votes_from_voters(voters, merged)
    for r in changed:
        print(f'    {r["action"]:6s} [{r["phash"]}] {r["old_name"] or "(none)"!r} → '
              f'{r["winner"]!r}   installs: {tally.get(r["phash"])}')

    if not args.apply:
        print('\nDRY-RUN — nothing written. Re-run with --apply.')
        return

    print('\nWriting knowledge.json (schema 4)...')
    if not admin_merge._hf_save_state(merged, sorted(processed), watermark,
                                      voters=voters, drain_contribs=None):
        print('ERROR — save failed.', file=sys.stderr)
        sys.exit(1)
    print('OK — voters written. Drain the counted backlog with drain_stale_staging.yml.')


if __name__ == '__main__':
    main()
