#!/usr/bin/env python3
"""
admin_merge.py — WARP Knowledge Base merger
============================================
Czyta wszystkie contributions z HF Dataset, robi majority-vote,
zapisuje wynik do knowledge.json.

Użycie:
    python admin_merge.py                    # dry-run (tylko raport)
    python admin_merge.py --apply            # zapisuje knowledge.json do HF
    python admin_merge.py --apply --min 1    # 1 głos wystarczy (domyślnie 2)
    python admin_merge.py --since 2026-03-01 # tylko contributions od tej daty

Zmienne środowiskowe (lub plik .env):
    HF_TOKEN     — HF write token
    HF_REPO_ID   — np. sets-sto/warp-knowledge
    ADMIN_KEY    — klucz admina (opcjonalny, do uderzenia w /admin/merge endpoint)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, date
from pathlib import Path

# ── Auto-restart w .venv jeśli potrzeba ───────────────────────────────────────

def _ensure_venv():
    """
    W pełni standalone — zero systemowego Pythona, zero systemowego pip.

    1. Jeśli już jesteśmy w lokalnym .venv → OK
    2. Jeśli .venv istnieje → restart w nim
    3. Jeśli brak .venv → uruchom setup.py który:
         - pobiera portable Python 3.12 do .python/
         - tworzy .venv z tego Pythona
         - instaluje requirements.txt (w tym huggingface-hub)
       Następnie restart w gotowym .venv
    """
    here     = Path(__file__).resolve().parent
    is_win   = sys.platform == 'win32'
    venv_py  = here / ('.venv/Scripts/python.exe' if is_win else '.venv/bin/python')
    setup_py = here / 'setup.py'

    # 1. Już jesteśmy w naszym .venv
    if venv_py.exists() and Path(sys.executable).resolve() == venv_py.resolve():
        return

    # 2. .venv istnieje — restart w nim
    if venv_py.exists():
        os.execv(str(venv_py), [str(venv_py)] + sys.argv)

    # 3. Brak .venv — uruchom setup.py (pobierze portable Python, zbuduje venv)
    if setup_py.exists():
        print('  → Brak .venv — uruchamiam setup.py (portable Python 3.12) ...')
        # setup.py jest interaktywny — uruchamiamy go i po zakończeniu
        # restart w nowo utworzonym .venv
        ret = subprocess.call([sys.executable, str(setup_py)])
        if ret != 0:
            print('ERROR: setup.py nie powiódł się.', file=sys.stderr)
            sys.exit(1)
        if venv_py.exists():
            os.execv(str(venv_py), [str(venv_py)] + sys.argv)
        else:
            print('ERROR: setup.py nie utworzył .venv.', file=sys.stderr)
            sys.exit(1)
    else:
        print('ERROR: brak setup.py — uruchom go ręcznie żeby skonfigurować środowisko.',
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
    """Pobiera wszystkie contribution JSON z HF Dataset."""
    if not HF_TOKEN or not HF_REPO_ID:
        print('ERROR: HF_TOKEN lub HF_REPO_ID nie ustawione', file=sys.stderr)
        sys.exit(1)
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print('ERROR: pip install huggingface-hub', file=sys.stderr)
        sys.exit(1)

    api   = HfApi(token=HF_TOKEN)
    files = list(api.list_repo_files(HF_REPO_ID, repo_type='dataset'))
    json_files = [
        f for f in files
        if f.startswith('contributions/') and f.endswith('.json')
    ]

    if since:
        json_files = [f for f in json_files if f.split('/')[1] >= since]

    print(f'Znaleziono {len(json_files)} plików contributions'
          + (f' od {since}' if since else ''))

    contribs = []
    for i, f in enumerate(json_files):
        try:
            local = hf_hub_download(HF_REPO_ID, f, repo_type='dataset', token=HF_TOKEN)
            data  = json.loads(Path(local).read_text(encoding='utf-8'))
            contribs.append(data)
        except Exception as e:
            print(f'  SKIP {f}: {e}')
        if (i + 1) % 50 == 0:
            print(f'  wczytano {i + 1}/{len(json_files)}...')

    return contribs


def _hf_load_knowledge() -> dict[str, str]:
    """Wczytuje aktualny knowledge.json z HF."""
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            HF_REPO_ID, 'knowledge.json',
            repo_type='dataset', token=HF_TOKEN or None,
        )
        data = json.loads(Path(local).read_text(encoding='utf-8'))
        return data.get('knowledge', data)
    except Exception as e:
        print(f'UWAGA: knowledge.json nie istnieje lub błąd ({e}) — zaczynam od zera')
        return {}


def _hf_save_knowledge(knowledge: dict[str, str]) -> bool:
    """Zapisuje knowledge.json do HF Dataset."""
    try:
        from huggingface_hub import HfApi
        api     = HfApi(token=HF_TOKEN)
        payload = json.dumps(
            {
                'knowledge':  knowledge,
                'updated_at': datetime.utcnow().isoformat() + 'Z',
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
                           f'({datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC)',
        )
        return True
    except Exception as e:
        print(f'ERROR: zapis knowledge.json: {e}', file=sys.stderr)
        return False


# ── Merge logic ────────────────────────────────────────────────────────────────

def merge(
    contribs:   list[dict],
    existing:   dict[str, str],
    min_votes:  int = 2,
    verbose:    bool = False,
) -> tuple[dict[str, str], list[dict]]:
    """
    Majority-vote merge.

    Zwraca (merged_knowledge, report_rows).
    report_rows — lista słowników do wyświetlenia.
    """
    # Grupuj po phash
    phash_votes: dict[str, Counter] = {}
    phash_meta:  dict[str, dict]    = {}   # phash → {total, confirmed, wrong_names}

    for c in contribs:
        if not isinstance(c, dict):
            continue
        ph   = c.get('phash', '').strip()
        name = c.get('item_name', '').strip()
        if not ph or not name:
            continue

        phash_votes.setdefault(ph, Counter())[name] += 1
        meta = phash_meta.setdefault(ph, {'total': 0, 'confirmed': 0, 'wrong': Counter()})
        meta['total'] += 1
        if c.get('confirmed'):
            meta['confirmed'] += 1
        wrong = c.get('wrong_name', '').strip()
        if wrong:
            meta['wrong'][wrong] += 1

    merged  = dict(existing)
    report  = []

    for ph, votes in sorted(phash_votes.items()):
        winner, count = votes.most_common(1)[0]
        meta          = phash_meta[ph]
        old_name      = existing.get(ph, '')

        # Próg: min_votes, chyba że phash jest nowy — wtedy 1 wystarczy
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
    old     = f" (było: {row['old_name'][:30]})" if row.get('old_name') and action == 'UPDATE' else ''
    wrong   = f" [błędne: {list(row.get('wrong', {}).keys())}]" if row.get('wrong') else ''
    print(f'  {color}{symbol} [{ph}] {winner!r:50s} {votes}/{total} głosów{old}{wrong}{reset}')


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='WARP Knowledge Base merger')
    parser.add_argument('--apply',   action='store_true',
                        help='Zapisz wynik do HF (domyślnie: dry-run)')
    parser.add_argument('--min',     type=int, default=2, metavar='N',
                        help='Minimalna liczba głosów (domyślnie: 2)')
    parser.add_argument('--since',   metavar='YYYY-MM-DD',
                        help='Uwzględnij contributions tylko od tej daty')
    parser.add_argument('--verbose', action='store_true',
                        help='Pokaż wszystkie wpisy (nie tylko zmiany)')
    parser.add_argument('--export',  metavar='FILE',
                        help='Zapisz wynikowy knowledge.json lokalnie')
    args = parser.parse_args()

    print('=' * 60)
    print(f'WARP Knowledge Merger')
    print(f'Repo:     {HF_REPO_ID}')
    print(f'Min głosów: {args.min}')
    print(f'Tryb:     {"APPLY" if args.apply else "DRY-RUN"}')
    if args.since:
        print(f'Od:       {args.since}')
    print('=' * 60)

    # 1. Wczytaj contributions
    contribs = _hf_list_contributions(since=args.since)
    if not contribs:
        print('Brak contributions — nic do zrobienia.')
        return

    confirmed = sum(1 for c in contribs if c.get('confirmed'))
    print(f'\nWczytano {len(contribs)} contributions ({confirmed} confirmed)\n')

    # 2. Wczytaj aktualny knowledge
    existing = _hf_load_knowledge()
    print(f'Aktualny knowledge.json: {len(existing)} wpisów\n')

    # 3. Merge
    merged, report = merge(contribs, existing, min_votes=args.min, verbose=args.verbose)

    # 4. Raport
    new_count     = sum(1 for r in report if r['action'] == 'NEW')
    update_count  = sum(1 for r in report if r['action'] == 'UPDATE')
    skip_count    = sum(1 for r in report if r['action'] == 'SKIP')
    unchanged     = sum(1 for r in report if r['action'] == 'unchanged')

    print(f'\n--- Podsumowanie ---')
    print(f'  ✓ Nowe:      {new_count}')
    print(f'  ↺ Zaktualizowane: {update_count}')
    print(f'  · Bez zmian: {unchanged}')
    print(f'  ✗ Pominięte (za mało głosów): {skip_count}')
    print(f'  Łącznie po merge: {len(merged)} wpisów')

    # 5. Lokalny export
    if args.export:
        Path(args.export).write_text(
            json.dumps({'knowledge': merged, 'updated_at': datetime.utcnow().isoformat() + 'Z'},
                       ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        print(f'\nZapisano lokalnie: {args.export}')

    # 6. Apply
    if args.apply:
        if new_count == 0 and update_count == 0:
            print('\nBrak zmian — knowledge.json nie zostanie nadpisany.')
            return
        print(f'\nZapisuję {len(merged)} wpisów do HF...')
        ok = _hf_save_knowledge(merged)
        if ok:
            print('OK — knowledge.json zaktualizowany na HF.')
        else:
            print('BŁĄD — zapis nieudany.', file=sys.stderr)
            sys.exit(1)
    else:
        print('\nDRY-RUN — użyj --apply żeby zapisać.')


if __name__ == '__main__':
    main()
