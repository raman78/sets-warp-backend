"""
hf_clone.py — shallow git clone of HF Hub repos
================================================
Replacement for huggingface_hub.snapshot_download in scripts that need a
local materialised view of a Hub repo's text files.

Why not snapshot_download?
    snapshot_download performs a per-file HEAD request to validate the
    local cache. Once a tree grows past ~2000 files, even at max_workers=4
    HF starts returning HTTP 429 and the run eventually dies on a
    ReadTimeout. A `git fetch` transfers the whole tree in a single
    network round-trip and never trips the limiter.

Token handling:
    The token is passed via `-c http.extraHeader=Authorization: Bearer …`
    so it never lands in `.git/config` (no plain-text token on disk).
    Argv exposure is irrelevant on ephemeral CI runners; GitHub Actions
    masks the secret in logs.

LFS:
    GIT_LFS_SKIP_SMUDGE=1 is forced — LFS pointer files are kept, blobs
    are not fetched. Callers that need binary LFS content must download
    those individually via hf_hub_download afterwards.

    Partial-clone (`--filter=blob:none`) is NOT used: HF's git server
    rejects promisor-remote fetches ("fatal: expected 'packfile'"), so
    callers that want to skip large binaries must rely on LFS instead.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def clone_hf_shallow(
    repo_id:   str,
    token:     str,
    repo_type: str = 'dataset',
) -> Path:
    """
    Shallow-clone (or fast-forward) a HF Hub repo to a stable cache dir.

    Returns the working-tree Path.
    """
    if not repo_id or not token:
        raise RuntimeError('clone_hf_shallow: repo_id and token are required')

    cache_root = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'warp-hf-clone'
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_root / repo_id.replace('/', '__')

    remote_url  = _remote_url(repo_id, repo_type)
    auth_header = f'Authorization: Bearer {token}'
    env = {
        **os.environ,
        'GIT_LFS_SKIP_SMUDGE': '1',
        'GIT_TERMINAL_PROMPT': '0',
    }

    def _git(args: list[str], cwd: Path | None = None) -> None:
        subprocess.run(['git', *args], check=True, env=env,
                       cwd=str(cwd) if cwd else None)

    # HF rate-limits shared CI IPs (429) even with a valid token. git surfaces
    # that as a non-zero exit, so retry the network step with exponential
    # backoff; everything else (bad token, missing repo) still fails fast on
    # the last attempt.
    def _git_with_retry(args: list[str], cwd: Path | None = None,
                        label: str = 'git') -> None:
        for attempt in range(5):
            try:
                _git(args, cwd=cwd)
                return
            except subprocess.CalledProcessError:
                if attempt == 4:
                    raise
                delay = 2 ** attempt
                print(f'  HF {label} failed (attempt {attempt + 1}/5) — '
                      f'sleeping {delay}s', file=sys.stderr)
                time.sleep(delay)

    if (cache_dir / '.git').exists():
        _git_with_retry(['-c', f'http.extraHeader={auth_header}',
                         'fetch', '--depth', '1', 'origin', 'HEAD'],
                        cwd=cache_dir, label='fetch')
        _git(['reset', '--hard', 'FETCH_HEAD'], cwd=cache_dir)
        return cache_dir

    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    _git_with_retry(['-c', f'http.extraHeader={auth_header}',
                     'clone', '--depth', '1', '--single-branch',
                     remote_url, str(cache_dir)],
                    label='clone')

    return cache_dir


def _remote_url(repo_id: str, repo_type: str) -> str:
    if repo_type == 'model':
        return f'https://huggingface.co/{repo_id}'
    if repo_type == 'dataset':
        return f'https://huggingface.co/datasets/{repo_id}'
    if repo_type == 'space':
        return f'https://huggingface.co/spaces/{repo_id}'
    raise ValueError(f'unknown repo_type: {repo_type!r}')
