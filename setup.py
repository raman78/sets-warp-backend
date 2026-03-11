#!/usr/bin/env python3
# setup.py — WARP Backend Konfigurator
# Uruchom: python setup.py
#
# 1. Pobiera portable Python 3.12 do .python/  (brak root, brak kompilacji)
# 2. Tworzy .venv z tym Pythonem
# 3. Instaluje zależności z requirements.txt
# 4. Pyta o HF_TOKEN → zapisuje do .env
# 5. Konfiguracja w pyproject.toml
# 6. Test połączenia z HF
# 7. Opcjonalnie uruchamia serwer

from __future__ import annotations
import os, platform, re, subprocess, sys, tarfile, tempfile, urllib.request, zipfile
from pathlib import Path

HERE       = Path(__file__).resolve().parent
PYTHON_DIR = HERE / '.python'
VENV_DIR   = HERE / '.venv'
IS_WINDOWS = sys.platform == 'win32'
IS_MAC     = sys.platform == 'darwin'
ARCH       = platform.machine().lower()

PBS_BASE    = 'https://github.com/astral-sh/python-build-standalone/releases/download'
PBS_TAG     = '20250317'
PBS_VERSION = '3.12.9'
PBS_TARGETS = {
    ('linux',   'x86_64'):  f'cpython-{PBS_VERSION}+{PBS_TAG}-x86_64-unknown-linux-gnu-install_only.tar.gz',
    ('linux',   'aarch64'): f'cpython-{PBS_VERSION}+{PBS_TAG}-aarch64-unknown-linux-gnu-install_only.tar.gz',
    ('darwin',  'x86_64'):  f'cpython-{PBS_VERSION}+{PBS_TAG}-x86_64-apple-darwin-install_only.tar.gz',
    ('darwin',  'arm64'):   f'cpython-{PBS_VERSION}+{PBS_TAG}-aarch64-apple-darwin-install_only.tar.gz',
    ('windows', 'x86_64'):  f'cpython-{PBS_VERSION}+{PBS_TAG}-x86_64-pc-windows-msvc-install_only.zip',
}

G='\033[92m'; Y='\033[93m'; R='\033[91m'; B='\033[94m'; W='\033[97m'; RS='\033[0m'
def ok(m):   print(f'{G}  ✓ {m}{RS}')
def err(m):  print(f'{R}  ✗ {m}{RS}')
def info(m): print(f'{B}  → {m}{RS}')
def warn(m): print(f'{Y}  ! {m}{RS}')
def hdr(m):  print(f'\n{W}{"─"*50}\n  {m}\n{"─"*50}{RS}')


def _platform_key():
    if IS_WINDOWS: return ('windows', 'x86_64')
    sn   = 'darwin' if IS_MAC else 'linux'
    arch = ('arm64' if ARCH in ('arm64','aarch64') and IS_MAC
            else 'aarch64' if ARCH == 'aarch64' else 'x86_64')
    return (sn, arch)

def _portable_exe() -> Path | None:
    if IS_WINDOWS:
        p = PYTHON_DIR / 'python' / 'python.exe'
        return p if p.exists() else None
    for c in [PYTHON_DIR/'python'/'bin'/f'python{PBS_VERSION[:4]}',
              PYTHON_DIR/'python'/'bin'/'python3',
              PYTHON_DIR/'python'/'bin'/'python']:
        if c.exists(): return c
    return None

def _venv_python() -> Path:
    return VENV_DIR/('Scripts/python.exe' if IS_WINDOWS else 'bin/python')

def _venv_pip() -> Path:
    return VENV_DIR/('Scripts/pip.exe' if IS_WINDOWS else 'bin/pip')

def _venv_uvicorn() -> Path:
    return VENV_DIR/('Scripts/uvicorn' if IS_WINDOWS else 'bin/uvicorn')

def _in_our_venv() -> bool:
    try: return Path(sys.executable).resolve() == _venv_python().resolve()
    except: return False

def _ask_yn(prompt, default=True) -> bool:
    ans = input(f'  {prompt}{" [T/n]" if default else " [t/N]"}: ').strip().lower()
    return default if not ans else ans in ('t','y','tak','yes')

def _load_env(path):
    if not path.exists(): return {}
    r = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k,_,v = line.partition('=')
            r[k.strip()] = v.strip().strip('"').strip("'")
    return r

def _save_env(path, vars):
    lines = ['# WARP Backend — NIE commituj!\n']
    lines += [f'{k}={v}' for k,v in vars.items()]
    path.write_text('\n'.join(lines)+'\n')

def _find_system_python() -> str:
    """Znajdź systemowego Pythona poza naszym venv."""
    import shutil
    # Szukaj w kolejności preferencji
    for candidate in ['python3', 'python']:
        p = shutil.which(candidate)
        if p and str(VENV_DIR) not in p:
            return p
    # Fallback: oryginalny Python którym uruchomiono setup.py
    # (może być w venv, ale to ostatnia deska ratunku)
    return sys.executable


def _read_toml(key):
    try:
        m = re.search(rf'{re.escape(key)}\s*=\s*["\']?([^"\'#\n]+)["\']?',
                      (HERE/'pyproject.toml').read_text())
        return m.group(1).strip() if m else None
    except: return None


def _venv_python_version() -> tuple[int,int] | None:
    """Zwróć (major, minor) Pythona w venv, lub None jeśli brak/błąd."""
    vp = _venv_python()
    if not vp.exists(): return None
    try:
        r = subprocess.run([str(vp), '-c',
            'import sys; print(sys.version_info.major, sys.version_info.minor)'],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            major, minor = map(int, r.stdout.strip().split())
            return (major, minor)
    except Exception:
        pass
    return None


def _cleanup_stale():
    """Usuń .venv i .python jeśli są niekompatybilne lub uszkodzone."""
    import shutil
    venv_ver = _venv_python_version()
    if venv_ver is not None and venv_ver == (3, 12):
        return   # wszystko OK

    if venv_ver is not None:
        warn(f'Stary venv ma Python {venv_ver[0]}.{venv_ver[1]} (potrzebny 3.12) — czyszczę ...')
    elif VENV_DIR.exists():
        warn('Uszkodzony lub niekompletny venv — czyszczę ...')

    for d in [VENV_DIR, PYTHON_DIR]:
        if d.exists():
            shutil.rmtree(d)
            ok(f'Usunięto: {d}')


def main():
    print(f'\n{W}WARP Knowledge Backend — Konfigurator{RS}\n{"═"*40}')

    # Nawet jeśli jesteśmy w venv — sprawdź czy to właściwa wersja
    if _in_our_venv():
        major, minor = sys.version_info[:2]
        if (major, minor) == (3, 12):
            ok(f'Uruchomiony w venv (Python {sys.version.split()[0]})')
            hdr('Konfiguracja HF Token');  _configure_env()
            hdr('Konfiguracja serwera');   _configure_toml()
            hdr('Test połączenia');        _test_hf()
            hdr('Uruchomienie');           _prompt_start()
            return
        else:
            warn(f'Venv ma Python {major}.{minor} zamiast 3.12 — czyszczę i reinstalluję ...')
            # Wyskocz z venv przed czyszczeniem — użyj systemowego Pythona
            import shutil
            for d in [VENV_DIR, PYTHON_DIR]:
                if d.exists():
                    shutil.rmtree(d)
                    ok(f'Usunięto: {d}')
            # Restart z systemowym Pythonem
            system_python = _find_system_python()
            info(f'Restartuję z systemowym Pythonem: {system_python}')
            os.execv(system_python, [system_python] + sys.argv)

    hdr('0. Sprawdzam środowisko');  _cleanup_stale()
    hdr('1. Portable Python');       _ensure_python()
    hdr('2. Wirtualne środowisko');  _ensure_venv()
    hdr('3. Zależności');            _install_deps()

    vp = _venv_python()
    if vp.exists() and Path(sys.executable).resolve() != vp.resolve():
        info('Restartuję w venv ...')
        os.execv(str(vp), [str(vp)] + sys.argv)

    hdr('4. HF Token');     _configure_env()
    hdr('5. Konfiguracja'); _configure_toml()
    hdr('6. Test HF');      _test_hf()
    hdr('7. Start');        _prompt_start()


def _ensure_python():
    if _portable_exe():
        ok(f'Portable Python już istnieje: {_portable_exe()}'); return

    key = _platform_key()
    fn  = PBS_TARGETS.get(key)
    if not fn:
        warn(f'Brak portable Pythona dla {key} — używam systemowego'); return

    url = f'{PBS_BASE}/{PBS_TAG}/{fn}'
    info(f'Pobieram Python {PBS_VERSION} (~65MB) ...')
    PYTHON_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix='.zip' if IS_WINDOWS else '.tar.gz')
    os.close(fd)
    try:
        last=[-1]
        def prog(c,b,t):
            if t>0:
                p=min(100,c*b*100//t)
                if p!=last[0]: last[0]=p; print(f'\r  {B}Pobieranie... {p}%{RS}',end='',flush=True)
        urllib.request.urlretrieve(url, tmp, reporthook=prog); print()
        info('Rozpakowuję ...')
        if fn.endswith('.tar.gz'):
            with tarfile.open(tmp,'r:gz') as t: t.extractall(PYTHON_DIR)
        else:
            with zipfile.ZipFile(tmp) as z: z.extractall(PYTHON_DIR)
    finally:
        try: os.unlink(tmp)
        except: pass

    exe = _portable_exe()
    if not exe: err('Nie znaleziono exe po rozpakowaniu'); sys.exit(1)
    if not IS_WINDOWS: exe.chmod(0o755)
    ok(f'Portable Python gotowy: {exe}')


def _ensure_venv():
    if VENV_DIR.exists(): ok('.venv już istnieje'); return
    exe = _portable_exe() or Path(sys.executable)
    info(f'Tworzę .venv ({exe}) ...')
    r = subprocess.run([str(exe), '-m', 'venv', str(VENV_DIR)])
    if r.returncode != 0: err('Błąd tworzenia venv'); sys.exit(1)
    ok('.venv utworzony')


def _install_deps():
    pip = _venv_pip()
    if not pip.exists(): err(f'pip nie znaleziony: {pip}'); sys.exit(1)
    info('Instaluję requirements.txt ...')
    r = subprocess.run([str(pip), 'install', '-r', str(HERE/'requirements.txt'), '--quiet'])
    if r.returncode == 0: ok('Zależności zainstalowane')
    else:
        err('Błąd instalacji')
        if not _ask_yn('Kontynuować?'): sys.exit(1)


def _configure_env():
    env_path = HERE / '.env'
    env      = _load_env(env_path)

    token = env.get('HF_TOKEN','')
    if token:
        warn(f'HF_TOKEN już ustawiony: {token[:8]}...{token[-4:]}')
        if _ask_yn('Zmienić?', default=False): token = ''
    if not token:
        print(f'\n  {B}https://huggingface.co/settings/tokens{RS}  → typ: Write')
        token = input('  Wklej HF_TOKEN: ').strip()
        if token: env['HF_TOKEN'] = token; ok('HF_TOKEN zapisany')
        else: warn('Pominięto')

    if not env.get('ADMIN_KEY'):
        import secrets
        key = secrets.token_urlsafe(32)
        env['ADMIN_KEY'] = key
        print(f'\n  {W}ADMIN_KEY (zapisz!): {key}{RS}')
        ok('ADMIN_KEY wygenerowany')

    _save_env(env_path, env)
    for k,v in env.items(): os.environ.setdefault(k,v)
    ok('.env zapisany')


def _configure_toml():
    path    = HERE / 'pyproject.toml'
    content = path.read_text()
    repo    = _read_toml('repo_id') or 'sets-sto/warp-knowledge'
    port    = _read_toml('port')    or '8000'
    info(f'HF repo: {repo}')
    info(f'Port:    {port}')
    if _ask_yn('Zmienić?', default=False):
        nr = input(f'  repo_id [{repo}]: ').strip() or repo
        np_ = input(f'  port [{port}]: ').strip() or port
        content = re.sub(r'(repo_id\s*=\s*)"[^"]+"', f'\\1"{nr}"', content)
        content = re.sub(r'(port\s*=\s*)\d+', f'\\g<1>{np_}', content)
        path.write_text(content); ok('pyproject.toml zaktualizowany')


def _test_hf():
    token = os.environ.get('HF_TOKEN','')
    repo  = _read_toml('repo_id') or 'sets-sto/warp-knowledge'
    if not token: warn('Brak HF_TOKEN — pomijam'); return
    info(f'Testuję {repo} ...')
    try:
        from huggingface_hub import HfApi
        info_obj = HfApi(token=token).repo_info(repo, repo_type='dataset')
        ok(f'HF OK: {info_obj.id}')
    except Exception as e:
        err(f'HF błąd: {e}')


def _prompt_start():
    port = _read_toml('port') or '8000'
    uvi  = _venv_uvicorn()
    print(f'\n  Start serwera:')
    print(f'  {G}{uvi} main:app --host 0.0.0.0 --port {port} --reload{RS}')
    print(f'  Docs: {B}http://localhost:{port}/docs{RS}')
    if _ask_yn('\nUruchomić teraz?', default=False):
        os.execv(str(uvi), [str(uvi),'main:app','--host','0.0.0.0','--port',port,'--reload'])


if __name__ == '__main__':
    main()
