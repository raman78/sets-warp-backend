"""A failed git command must not carry the token in its message.

The token is passed to git in argv (as a header config), and
`CalledProcessError`'s default message prints the whole command line. On
2026-09-03 a clone failed during a local run and put a live token into a
terminal transcript, which then had to be rotated.

`hf_clone`'s docstring had said argv exposure was "irrelevant on ephemeral CI
runners" because GitHub masks its own secrets. These tools also run in a
maintainer's shell, where nothing masks anything.

Offline: no HF, no network — git is a stub that fails.
"""
from __future__ import annotations

import subprocess

import pytest

import hf_clone

TOKEN = 'hf_notarealtokenbutlongenough123456'


def test_a_failure_does_not_print_the_token(monkeypatch, tmp_path):
    import time as _time

    def _boom(cmd, **kw):
        raise subprocess.CalledProcessError(128, cmd)

    monkeypatch.setattr(subprocess, 'run', _boom)
    # The network step retries with a 60-480 s backoff; without this the test
    # sits through ~22 minutes of it.
    monkeypatch.setattr(_time, 'sleep', lambda *_a: None)
    monkeypatch.setenv('HF_HOME', str(tmp_path))

    with pytest.raises(subprocess.CalledProcessError) as exc:
        hf_clone.clone_hf_shallow('some/repo', TOKEN, repo_type='dataset')

    assert TOKEN not in str(exc.value)
    assert TOKEN not in ' '.join(str(a) for a in exc.value.cmd)


def test_the_redaction_leaves_everything_else_readable():
    """A message with the token removed still has to say what failed."""
    line = f'http.extraHeader=Authorization: Bearer {TOKEN}'

    out = hf_clone._redact(line, TOKEN)

    assert TOKEN not in out
    assert 'http.extraHeader' in out


def test_redaction_is_a_no_op_without_a_token():
    assert hf_clone._redact('git fetch origin', '') == 'git fetch origin'
