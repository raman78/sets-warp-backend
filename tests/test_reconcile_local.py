"""Putting this machine's store beside the published dataset.

Every upload fault found so far was silent in the same way: the client
believed it had shared something the dataset does not have, and nothing put
the two side by side. Each was found by hand, months late, because a count
looked wrong.

Run standalone:
    python -m pytest tests/test_reconcile_local.py -v
"""
from __future__ import annotations

import json

import admin_reconcile_local as rec


# ── The three states ───────────────────────────────────────────────────────

def test_a_matching_pair_is_not_reported():
    v = rec.compare({'aa': 'BOFFS'}, {'aa': 'BOFFS'})

    assert v == {'missing': [], 'mislabelled': [], 'withdrawn': []}


def test_something_confirmed_here_and_absent_there_is_missing():
    v = rec.compare({'aa': 'DISCARD'}, {})

    assert v['missing'] == ['aa']


def test_a_different_label_is_reported_with_both_sides():
    """The fault this exists for: the file was sent, the correction was not,
    so the dataset holds the first label it was given."""
    v = rec.compare({'aa': 'SPACE_BOFFS'}, {'aa': 'BOFFS'})

    assert v['mislabelled'] == [('aa', 'SPACE_BOFFS', 'BOFFS')]
    assert v['missing'] == []


def test_something_published_but_gone_here_is_withdrawn():
    """Not a fault on its own — a maintainer rejection removes it here."""
    v = rec.compare({}, {'aa': 'BOFFS'})

    assert v['withdrawn'] == ['aa']


# ── Reading each side ──────────────────────────────────────────────────────

def test_the_unclassified_folder_is_not_compared(tmp_path):
    """`UNKNOWN` is a sentinel the backend refuses by design, so counting it
    would report a backlog nobody can clear."""
    d = tmp_path / 'screen_types' / rec.UNCLASSIFIED
    d.mkdir(parents=True)
    (d / 'a.png').write_bytes(b'\x89PNG' + b'\x00' * 50)

    assert rec.local_screens(tmp_path) == {}


def test_a_classified_screenshot_is_read_with_its_type(tmp_path):
    d = tmp_path / 'screen_types' / 'SPACE_BOFFS'
    d.mkdir(parents=True)
    (d / 'a.png').write_bytes(b'\x89PNG' + b'\x00' * 50)

    assert list(rec.local_screens(tmp_path).values()) == ['SPACE_BOFFS']


def test_the_local_hash_is_truncated_like_the_client(tmp_path):
    """The client truncates to 32 hex chars and the dataset is keyed on that.
    Comparing full digests finds nothing at all — and an empty result reads
    as 'everything agrees', which is the worst way for this to fail."""
    d = tmp_path / 'screen_types' / 'BOFFS'
    d.mkdir(parents=True)
    (d / 'a.png').write_bytes(b'\x89PNG')

    sha = next(iter(rec.local_screens(tmp_path)))

    # 32 literally, not `rec.SHA_LEN` — asserting against the constant the
    # code uses compares it with itself and passes whatever it is set to.
    assert len(sha) == 32


def test_published_screens_are_read_from_the_merged_metadata(tmp_path):
    p = tmp_path / 'data' / 'screen_types'
    p.mkdir(parents=True)
    (p / 'metadata.jsonl').write_text(
        json.dumps({'sha': 'aa', 'type': 'BOFFS'}) + '\n'
        + '{ broken\n'
        + json.dumps({'sha': 'bb', 'type': 'TRAITS'}) + '\n',
        encoding='utf-8')

    assert rec.published_screens(tmp_path) == {'aa': 'BOFFS', 'bb': 'TRAITS'}


def test_a_missing_dataset_file_is_not_an_error(tmp_path):
    assert rec.published_screens(tmp_path) == {}
    assert rec.published_crops(tmp_path) == {}


def test_a_crop_label_comes_from_the_annotations_not_the_filename(tmp_path):
    """The filename carries the label the file had when it was written, so a
    correction made later would be invisible exactly where it matters."""
    (tmp_path / 'crops').mkdir()
    png = tmp_path / 'crops' / 'boff_tactical__old_name__abc123def456.png'
    png.write_bytes(b'\x89PNG' + b'\x00' * 50)
    (tmp_path / 'annotations.json').write_text(json.dumps({
        'key': {'filename': 's.png', 'annotations': [
            {'ann_id': 'abc123def456', 'slot': 'Boff Tactical',
             'name': 'Corrected Name'}]}}), encoding='utf-8')

    assert list(rec.local_crops(tmp_path).values()) == \
        ['Boff Tactical|Corrected Name']
