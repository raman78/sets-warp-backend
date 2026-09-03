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

    It is still in argv, and that is not harmless: these tools are run
    locally by the maintainer as well as on CI, and `CalledProcessError`
    prints the whole command line. A failed clone put a live token into a
    terminal transcript on 2026-09-03. Every failure from here is re-raised
    with the token replaced, so the value cannot travel in an error message.
    GitHub masks its own secrets in Actions logs; nothing masks a local
    shell.

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


def _redact(text: str, token: str) -> str:
    """Replace a token wherever it appears in a string."""
    return text.replace(token, '***') if token and isinstance(text, str) else text


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

    # Pre-flight: hit the API endpoint via huggingface_hub. Since 1.2.0 the
    # SDK parses the IETF RateLimit-Reset header on a 429 and waits the
    # exact reset window — much smarter than git, which only sees a bare
    # "fatal: ... error 429". If repo_info succeeds the 5-minute window is
    # fresh and the subsequent clone/fetch will not be throttled.
    try:
        from huggingface_hub import HfApi
        HfApi(token=token).repo_info(repo_id=repo_id, repo_type=repo_type)
    except Exception as e:
        # Don't fail outright — the legacy retry below still gets a chance,
        # and a missing repo / bad token will surface there with the same
        # error git produces anyway.
        print(f'  pre-flight repo_info({repo_id}) warning: {e}',
              file=sys.stderr, flush=True)

    remote_url  = _remote_url(repo_id, repo_type)
    auth_header = f'Authorization: Bearer {token}'
    env = {
        **os.environ,
        'GIT_LFS_SKIP_SMUDGE': '1',
        'GIT_TERMINAL_PROMPT': '0',
    }

    def _git(args: list[str], cwd: Path | None = None) -> None:
        try:
            subprocess.run(['git', *args], check=True, env=env,
                           cwd=str(cwd) if cwd else None)
        except subprocess.CalledProcessError as exc:
            # The token is in argv, and the default message prints argv.
            raise subprocess.CalledProcessError(
                exc.returncode,
                [_redact(a, token) for a in exc.cmd],
                output=exc.output, stderr=exc.stderr,
            ) from None

    # HF rate-limits shared CI IPs (429) even with a valid token. git surfaces
    # that as a non-zero exit, so retry the network step with exponential
    # backoff; everything else (bad token, missing repo) wastes a few cycles
    # before failing on the last attempt.
    #
    # Schedule must outlive HF's 60-120 s throttle window — see hf_retry.py.
    import random
    _delays = (60, 120, 240, 480, 480)
    def _git_with_retry(args: list[str], cwd: Path | None = None,
                        label: str = 'git') -> None:
        for attempt, base in enumerate(_delays):
            try:
                _git(args, cwd=cwd)
                return
            except subprocess.CalledProcessError:
                if attempt == len(_delays) - 1:
                    raise
                delay = max(1.0, base + random.uniform(-15, 15))
                print(f'  HF {label} failed (attempt {attempt + 1}/{len(_delays)}) — '
                      f'sleeping {delay:.0f}s', file=sys.stderr, flush=True)
                time.sleep(delay)

    # Fast-forward an existing cache — one attempt, not five.
    #
    # The retry schedule below exists for HF throttling, where waiting is the
    # cure. A fetch into a wedged shallow clone is not that: it fails the same
    # way every time, so the backoff spends ~22 minutes arriving at the answer
    # it had after the first second. Seen on 2026-09-04 as
    # `fatal: expected 'acknowledgments'`, cured instantly by deleting the
    # directory — which is exactly what the fresh-clone path below does.
    #
    # So the cheap path gets one shot and any failure falls through to the
    # path that works from nothing. A genuine 429 is not lost: the clone
    # keeps the full retry schedule.
    if (cache_dir / '.git').exists():
        try:
            _git(['-c', f'http.extraHeader={auth_header}',
                  'fetch', '--depth', '1', 'origin', 'HEAD'], cwd=cache_dir)
            _git(['reset', '--hard', 'FETCH_HEAD'], cwd=cache_dir)
            return cache_dir
        except subprocess.CalledProcessError:
            print('  HF fetch into the cached clone failed — discarding it '
                  'and cloning fresh', file=sys.stderr, flush=True)

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
