"""A photograph of a text line must not become a class of the icon models.

`data/annotations.jsonl` carries every confirmed crop, including the two slots
whose crop is a band of text rather than an icon: `Ship Type` (the ship's class
line) and `Ship Tier` (the `[T6-X2]` badge). Their labels are ship names and
tier strings.

`read_curated_crops` feeds both `admin_train` and `admin_train_metric`, and it
used to build `labels[sha] = name` without ever reading `slot`. Measured
2026-09-05 against the *shipped* models: 49 of the 3189 classes in
`label_map.json` and `embedder_label_map.json` were ship names and tier
strings — `T6`, `T6-X2`, `Kor Bird-of-Prey`, `Terran Lexington Dreadnought
Cruiser` — every one of them occurring only on text rows. The icon matcher
could answer `T6-X2` for an equipment slot.

Every other tool that reasons about labels already knew this
(`admin_clean_labels`, `democratic_merge_screens`, `admin_reject_crops`); this
reader was the one that did not.

Offline: `hf_hub_download` is replaced with a local file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import admin_train


ROWS = [
    # Real icons — these must survive.
    {'crop_sha256': 'a1', 'name': 'Console - Tactical - Phaser Relay',
     'slot': 'Tactical Consoles', 'votes': 3},
    {'crop_sha256': 'a2', 'name': 'Phaser Beam Array',
     'slot': 'Fore Weapons', 'votes': 5},
    {'crop_sha256': 'a3', 'name': '__empty__',
     'slot': 'Devices', 'votes': 2},
    # Text bands — these must not.
    {'crop_sha256': 'b1', 'name': 'T6-X2', 'slot': 'Ship Tier', 'votes': 4},
    {'crop_sha256': 'b2', 'name': 'Kor Bird-of-Prey',
     'slot': 'Ship Type', 'votes': 6},
    # The exact shape that reached the published model: a class name recorded
    # under the tier slot, because the two rows shared one bbox and therefore
    # one crop hash.
    {'crop_sha256': 'b3', 'name': 'Fleet Yamaguchi Support Cruiser',
     'slot': 'Ship Tier', 'votes': 5},
]


@pytest.fixture
def labels_and_votes(tmp_path, monkeypatch, capsys):
    jsonl = tmp_path / 'annotations.jsonl'
    jsonl.write_text('\n'.join(json.dumps(r) for r in ROWS), encoding='utf-8')

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download',
                        lambda *a, **kw: str(jsonl))

    labels, votes = admin_train.read_curated_crops()
    return labels, votes, capsys.readouterr().out


def test_icon_crops_are_kept(labels_and_votes):
    labels, _votes, _out = labels_and_votes
    assert labels == {
        'a1': 'Console - Tactical - Phaser Relay',
        'a2': 'Phaser Beam Array',
        'a3': '__empty__',
    }


def test_a_tier_badge_is_not_an_icon_class(labels_and_votes):
    labels, _votes, _out = labels_and_votes
    assert 'T6-X2' not in labels.values()


def test_a_ship_class_line_is_not_an_icon_class(labels_and_votes):
    labels, _votes, _out = labels_and_votes
    assert 'Kor Bird-of-Prey' not in labels.values()
    assert 'Fleet Yamaguchi Support Cruiser' not in labels.values()


def test_virtual_labels_still_survive(labels_and_votes):
    """__empty__ is not text — it is an ordinary class of the icon models."""
    labels, _votes, _out = labels_and_votes
    assert labels.get('a3') == '__empty__'


def test_votes_are_dropped_with_their_label(labels_and_votes):
    """A vote count left behind for a skipped crop would make the min-votes
    filter operate on a crop that is no longer in the training set."""
    labels, votes, _out = labels_and_votes
    assert set(votes) == set(labels)


def test_the_skip_is_reported_not_silent(labels_and_votes):
    """Rule: a rejection is surfaced. A count that climbs is how anyone
    learns that clients are still uploading text crops."""
    _labels, _votes, out = labels_and_votes
    assert 'skipped 3 text-slot record' in out


def test_ship_name_counts_as_text_too():
    """`Ship Name` is anchor-internal and never emitted as a slot today, but
    an old annotation can still carry it, and it is a text band either way."""
    assert 'Ship Name' in admin_train.TEXT_LEARNING_SLOTS


def test_both_trainers_share_this_reader():
    """One fix has to cover the classifier and the embedder. If the metric
    trainer ever grows its own reader, this test says so."""
    src = (Path(admin_train.__file__).parent / 'admin_train_metric.py').read_text()
    assert 'from admin_train import' in src
    assert 'read_curated_crops' in src
