"""No test may write to the Hugging Face Hub.

The admin tools read HF_TOKEN from `.env`, so on the maintainer's machine a
test that reaches a real write path writes to production. That happened on
2026-10-01: a test of admin_merge's schema-3 refusal, run with the refusal
disabled to prove the test bites, fell through to the real listing and save,
overwrote knowledge.json and deleted 1,024 contribution files before a chunk
failed. Restored from history the same hour.

Every write method on `HfApi` raises here. A test that wants to observe a
write replaces `HfApi` (or the tool's commit helper) with its own recorder,
as several already do, and is unaffected.
"""
from __future__ import annotations

import pytest

_WRITES = ('create_commit', 'upload_file', 'upload_folder', 'delete_file',
           'delete_files', 'delete_folder', 'create_repo', 'delete_repo')


@pytest.fixture(autouse=True)
def _no_hub_writes(monkeypatch):
    import huggingface_hub

    def _refuse(name):
        def _raise(*a, **k):
            raise RuntimeError(f'test tried to write to the HF Hub: HfApi.{name}')
        return _raise

    for name in _WRITES:
        monkeypatch.setattr(huggingface_hub.HfApi, name, _refuse(name), raising=False)
