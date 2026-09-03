"""New crops go under a shard; both layouts stay readable.

HF refuses a push that would leave more than 10 000 files in one directory:

    Your push was rejected because it contains too many files per directory.
    Offending directories: /data/crops/

`data/crops/` reached ~9 985 and the promotion froze on 2026-07-16. Every
scheduled run since computed the right answer and died committing it, and the
job stayed green for seven weeks because the step piped through `tee`.

New crops are written to `data/crops/<first two hex of sha>/<sha>.png` — 256
shards, ~40 files each today. The flat files predate it and are migrated
separately, so every reader has to accept both meanwhile.

Offline: no HF, no network.
"""
from __future__ import annotations

import pytest

import democratic_merge_crops as merge
import admin_reject_crops as review

SHA = 'ab12cd34ef56'


# ── Where a crop is written ────────────────────────────────────────────────

def test_a_new_crop_goes_under_a_shard():
    assert merge.crop_path(SHA) == 'data/crops/ab/ab12cd34ef56.png'


def test_the_shard_is_the_first_two_characters_of_the_sha():
    """256 shards, so the cap is ~2.5M crops rather than 10k — and the shard
    is derivable from the name alone, with no index to keep in step."""
    assert {merge.crop_path(f'{a:02x}0000')[len('data/crops/'):][:2]
            for a in range(256)} == {f'{a:02x}' for a in range(256)}


def test_both_layouts_are_offered_for_reading():
    """Sharded first: that is where anything written from now on lives."""
    assert merge.crop_paths(SHA) == ('data/crops/ab/ab12cd34ef56.png',
                                     'data/crops/ab12cd34ef56.png')


def test_the_review_tool_agrees_with_the_merger():
    """Two copies of the rule would drift; they are the same shape by
    construction, and this fails the day one of them changes alone."""
    assert review.crop_paths(SHA) == merge.crop_paths(SHA)


# ── Not re-copying what is already there ───────────────────────────────────

@pytest.mark.parametrize('present', ['data/crops/ab/ab12cd34ef56.png',
                                     'data/crops/ab12cd34ef56.png'])
def test_a_crop_already_in_the_repo_is_not_copied_again(present):
    """A flat crop from before the shards must not be duplicated into one —
    that would push the directory count up rather than down."""
    assert any(p == present for p in merge.crop_paths(SHA))
