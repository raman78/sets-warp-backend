#!/usr/bin/env python3
# setup.py
# WARP Backend Configurator
#
# Uruchom: python setup.py
#
# Co robi:
#   1. Instaluje zależności (pip install -e .)
#   2. Pyta o HF_TOKEN i zapisuje do .env
#   3. Pozwala zmienić konfigurację w pyproject.toml
#   4. Testuje połączenie z HF i backendem
#   5. Opcjonalnie uruchamia serwer lokalnie

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

# ── ANSI colors ────────────────────────────────────────────────────────────────
G  = '\033[92m'   # green
Y  = '\033[93m'   # yellow
R  = '\033[91m'   # red
B  = '\033[94m'   # blue
W  = '\033[97m'   # white bold
RS = '\033[0m'    # reset

def ok(msg):  print(f'{G}  ✓ {msg}{RS}')
def err(msg): print(f'{R}  ✗ {msg}{RS}')
def info(msg):print(f'{B}  → {msg}{RS}')
def warn(msg):print(f'{Y}  ! {msg}{RS}')
def hdr(msg): print(f'\n{W}{'─'*50}\n  {msg}\n{'─'*50}{RS}')


def main():
    print(f'\n{W}WARP Knowledge Backend — Konfigurator{RS}')
    print(f'{"═"*40}')

    # ── Krok 1: Instalacja zależności ──────────────────────────────────────────
    hdr('1. Instalacja zależności')
    _install_deps()

    # ── Krok 2: Konfiguracja .env ──────────────────────────────────────────────
    hdr('2. Konfiguracja HF Token')
    _configure_env()

    # ── Krok 3: Konfiguracja pyproject.toml ───────────────────────────────────
    hdr('3. Konfiguracja serwera')
    _configure_toml()

    # ── Krok 4: Test połączenia ────────────────────────────────────────────────
    hdr('4. Test połączenia')
    _test_connections()

    # ── Krok 5: Uruchomienie ───────────────────────────────────────────────────
    hdr('5. Uruchomienie')
    _prompt_start()


# ── Step implementations ───────────────────────────────────────────────────────

def _install_deps():
    info('Instaluję zależności z pyproject.toml ...')
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-e', '.', '--quiet'],
        cwd=HERE
    )
    if result.returncode == 0:
        ok('Zależności zainstalowane')
    else:
        err('Błąd instalacji — sprawdź output powyżej')
        if not _ask_yn('Kontynuować mimo błędu?'):
            sys.exit(1)


def _configure_env():
    env_path = HERE / '.env'
    env_vars = _load_env(env_path)

    # HF_TOKEN
    current_token = env_vars.get('HF_TOKEN', '')
    if current_token:
        masked = current_token[:8] + '...' + current_token[-4:]
        warn(f'HF_TOKEN już ustawiony: {masked}')
        if not _ask_yn('Zmienić?', default=False):
            pass
        else:
            current_token = ''

    if not current_token:
        print(f'\n  Wejdź na: {B}https://huggingface.co/settings/tokens{RS}')
        print(f'  Stwórz token typu {W}Write{RS}')
        token = input('  Wklej HF_TOKEN: ').strip()
        if token:
            env_vars['HF_TOKEN'] = token
            ok('HF_TOKEN zapisany')
        else:
            warn('Pominięto — backend nie będzie mógł zapisywać do HF')

    # ADMIN_KEY
    if not env_vars.get('ADMIN_KEY'):
        import secrets
        admin_key = secrets.token_urlsafe(32)
        env_vars['ADMIN_KEY'] = admin_key
        ok(f'ADMIN_KEY wygenerowany: {admin_key[:16]}...')
        warn('Zapisz ten klucz! Potrzebny do wywołania /admin/merge')

    _save_env(env_path, env_vars)
    ok(f'.env zapisany: {env_path}')

    # Załaduj do os.environ dla dalszych kroków
    for k, v in env_vars.items():
        os.environ.setdefault(k, v)


def _configure_toml():
    toml_path = HERE / 'pyproject.toml'
    content   = toml_path.read_text(encoding='utf-8')

    # Pokaż aktualne wartości
    repo_match = re.search(r'repo_id\s*=\s*"([^"]+)"', content)
    port_match  = re.search(r'port\s*=\s*(\d+)', content)
    current_repo = repo_match.group(1) if repo_match else 'sets-sto/warp-knowledge'
    current_port = port_match.group(1) if port_match else '8000'

    info(f'HF repo:  {current_repo}')
    info(f'Port:     {current_port}')

    if _ask_yn('Zmienić konfigurację?', default=False):
        new_repo = input(f'  HF repo_id [{current_repo}]: ').strip() or current_repo
        new_port = input(f'  Port [{current_port}]: ').strip() or current_port

        content = re.sub(r'(repo_id\s*=\s*)"[^"]+"', f'\\1"{new_repo}"', content)
        content = re.sub(r'(port\s*=\s*)\d+', f'\\g<1>{new_port}', content)
        toml_path.write_text(content, encoding='utf-8')
        ok('pyproject.toml zaktualizowany')

        # Zsynchronizuj z main.py (env override i tak wygrywa na Render)
        os.environ['HF_REPO_ID'] = new_repo


def _test_connections():
    # Test HF
    hf_token  = os.environ.get('HF_TOKEN', '')
    hf_repo   = _read_toml_value('repo_id') or os.environ.get('HF_REPO_ID', 'sets-sto/warp-knowledge')

    if hf_token:
        info(f'Testuję dostęp do HF repo: {hf_repo} ...')
        try:
            from huggingface_hub import HfApi
            api  = HfApi(token=hf_token)
            info_obj = api.repo_info(hf_repo, repo_type='dataset')
            ok(f'HF repo dostępny: {info_obj.id}')
        except Exception as e:
            err(f'HF niedostępny: {e}')
    else:
        warn('Brak HF_TOKEN — pomijam test HF')

    # Test lokalny (opcjonalny — serwer musi być uruchomiony)
    info('Testuję lokalny serwer (http://localhost:8000/health) ...')
    try:
        import urllib.request
        with urllib.request.urlopen('http://localhost:8000/health', timeout=3) as r:
            import json
            data = json.loads(r.read())
            ok(f'Lokalny serwer odpowiada: {data}')
    except Exception:
        warn('Lokalny serwer nie odpowiada (normalnie jeśli jeszcze nie uruchomiony)')


def _prompt_start():
    port = _read_toml_value('port') or '8000'
    print(f'\n  Aby uruchomić serwer lokalnie:')
    print(f'  {G}uvicorn main:app --host 0.0.0.0 --port {port} --reload{RS}')
    print(f'\n  Dokumentacja API:')
    print(f'  {B}http://localhost:{port}/docs{RS}')

    if _ask_yn('\nUruchomić teraz?', default=False):
        os.execv(sys.executable, [
            sys.executable, '-m', 'uvicorn',
            'main:app',
            '--host', '0.0.0.0',
            '--port', port,
            '--reload'
        ])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ask_yn(prompt: str, default: bool = True) -> bool:
    suffix = ' [T/n]' if default else ' [t/N]'
    ans = input(f'  {prompt}{suffix}: ').strip().lower()
    if not ans:
        return default
    return ans in ('t', 'y', 'tak', 'yes')


def _load_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, _, v = line.partition('=')
            result[k.strip()] = v.strip().strip('"').strip("'")
    return result


def _save_env(path: Path, vars: dict[str, str]) -> None:
    lines = ['# WARP Backend — zmienne środowiskowe', '# NIE commituj tego pliku!\n']
    for k, v in vars.items():
        lines.append(f'{k}={v}')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _read_toml_value(key: str) -> str | None:
    try:
        content = (HERE / 'pyproject.toml').read_text(encoding='utf-8')
        m = re.search(rf'{re.escape(key)}\s*=\s*["\']?([^"\'#\n]+)["\']?', content)
        return m.group(1).strip() if m else None
    except Exception:
        return None


if __name__ == '__main__':
    main()
