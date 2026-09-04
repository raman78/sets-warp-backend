"""The global SETS-gap ledger: one install, one vote.

The whole point of collecting these centrally is to answer "how many separate
installs hit this item". Anything that lets a single install count more than
once turns the number into noise, and a number nobody can trust is worse than
no number — it would be quoted in a request to the wiki or to SETS.

Run standalone:
    python -m pytest tests/test_sets_gaps.py -v
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

import admin_sets_gaps_report as report


def _ledger(install: str, *items) -> dict:
    return {'install_id': install,
            'items': [{'name': n, 'reason': r, 'slots': list(s)}
                      for n, r, s in items]}


def test_two_installs_reporting_the_same_item_count_twice():
    agg = report._tally([
        _ledger('a' * 8, ('Colony Rifle', 'missing-from-cargo', [])),
        _ledger('b' * 8, ('Colony Rifle', 'missing-from-cargo', [])),
    ])

    assert agg[('missing-from-cargo', 'Colony Rifle')]['installs'] == 2


def test_one_install_listing_an_item_twice_counts_once():
    """A duplicated entry must not read as two users wanting the same fix."""
    agg = report._tally([
        _ledger('a' * 8,
                ('Colony Rifle', 'missing-from-cargo', []),
                ('Colony Rifle', 'missing-from-cargo', [])),
    ])

    assert agg[('missing-from-cargo', 'Colony Rifle')]['installs'] == 1


def test_the_two_reasons_are_counted_separately():
    """They are requests to two different projects."""
    agg = report._tally([
        _ledger('a' * 8,
                ('Odd Item', 'missing-from-cargo', []),
                ('Odd Item', 'sets-loader-skips', [])),
    ])

    assert agg[('missing-from-cargo', 'Odd Item')]['installs'] == 1
    assert agg[('sets-loader-skips', 'Odd Item')]['installs'] == 1


def test_slots_from_different_installs_are_unioned():
    agg = report._tally([
        _ledger('a' * 8, ('Colony Rifle', 'missing-from-cargo',
                          ['ground/ground_weapons'])),
        _ledger('b' * 8, ('Colony Rifle', 'missing-from-cargo',
                          ['ground/kit_modules'])),
    ])

    assert agg[('missing-from-cargo', 'Colony Rifle')]['slots'] == {
        'ground/ground_weapons', 'ground/kit_modules'}


def test_a_nameless_or_reasonless_row_is_ignored():
    agg = report._tally([{'install_id': 'a' * 8, 'items': [
        {'name': '', 'reason': 'missing-from-cargo'},
        {'name': 'X', 'reason': ''},
    ]}])

    assert agg == {}


def test_a_corrupt_ledger_file_does_not_stop_the_report(tmp_path, capsys):
    """One bad file must not hide every other install's data."""
    d = tmp_path / 'sets_gaps'
    d.mkdir()
    (d / 'aaaaaaaa.json').write_text('{ broken', encoding='utf-8')
    (d / 'bbbbbbbb.json').write_text(
        json.dumps(_ledger('b' * 8, ('Colony Rifle', 'missing-from-cargo', []))),
        encoding='utf-8')

    loaded = report._load(tmp_path)

    assert len(loaded) == 1
    assert 'skipped' in capsys.readouterr().err


def test_no_directory_yet_is_not_an_error(tmp_path):
    """Before the first upload the folder does not exist."""
    assert report._load(tmp_path) == []


# ── Stale installs ─────────────────────────────────────────────────────────
#
# A live install pushes daily. One that is gone leaves its last file behind
# forever, and counting it would let uninstalled copies keep voting — the
# count is the whole argument, so it has to age out on its own.

def _at(days_ago: int) -> dict:
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return {'install_id': 'a' * 8, 'items': [],
            'updated_at': ts.isoformat(timespec='seconds').replace('+00:00', 'Z')}


def test_a_recently_refreshed_ledger_counts():
    assert not report._is_stale(_at(1), datetime.now(timezone.utc), 90)


def test_an_install_gone_for_months_stops_counting():
    assert report._is_stale(_at(200), datetime.now(timezone.utc), 90)


def test_a_ledger_with_no_timestamp_is_treated_as_stale():
    """It cannot be shown to be current, and over-counting is the failure
    that matters here."""
    assert report._is_stale({'items': []}, datetime.now(timezone.utc), 90)


def test_an_unparseable_timestamp_does_not_crash_the_report():
    assert report._is_stale({'updated_at': 'last Tuesday'},
                            datetime.now(timezone.utc), 90)


# ── The endpoint ───────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient
    import main

    uploaded: dict[str, bytes] = {}
    monkeypatch.setattr(main, '_hf_upload_files',
                        lambda files, message, repo_id: uploaded.update(files) or True)

    async def _ok(*a, **k):
        return True
    monkeypatch.setattr(main, '_check_and_increment_rate_limit', _ok)

    c = TestClient(main.app)
    c.uploaded = uploaded
    return c


def _post(client, **kw):
    body = {'install_id': 'a' * 12, 'items': []}
    body.update(kw)
    return client.post('/upload/sets-gaps', json=body)


def test_an_upload_is_stored_under_the_install_id(client):
    r = _post(client, items=[{'name': 'Colony Rifle',
                              'reason': 'missing-from-cargo',
                              'slots': ['ground/ground_weapons']}])

    assert r.status_code == 200
    assert r.json()['accepted'] == 1
    assert f'sets_gaps/{"a" * 12}.json' in client.uploaded


def test_an_unknown_reason_is_rejected_not_filed_under_the_wrong_one(client):
    """A client/server version mismatch must not quietly land in one bucket —
    the two reasons go to two different projects."""
    r = _post(client, items=[{'name': 'X', 'reason': 'something-new'}])

    assert r.json()['accepted'] == 0
    assert r.json()['rejected'] == 1


def test_a_duplicate_within_one_request_is_collapsed(client):
    r = _post(client, items=[
        {'name': 'Colony Rifle', 'reason': 'missing-from-cargo'},
        {'name': 'Colony Rifle', 'reason': 'missing-from-cargo'},
    ])

    stored = json.loads(client.uploaded[f'sets_gaps/{"a" * 12}.json'])
    assert r.json()['accepted'] == 1
    assert len(stored['items']) == 1


def test_an_empty_ledger_is_accepted_so_entries_can_expire(client):
    """Once the wiki adds the cargo row, the item stops being reported and
    must drop out — which only works if an empty upload overwrites."""
    r = _post(client, items=[])

    assert r.status_code == 200
    assert json.loads(client.uploaded[f'sets_gaps/{"a" * 12}.json'])['items'] == []


def test_a_bad_install_id_is_refused(client):
    r = _post(client, install_id='../../etc/passwd')

    assert r.status_code == 400


def test_storage_failure_is_reported_not_silently_dropped(client, monkeypatch):
    import main
    monkeypatch.setattr(main, '_hf_upload_files',
                        lambda files, message, repo_id: False)

    r = _post(client, items=[{'name': 'X', 'reason': 'missing-from-cargo'}])

    assert r.status_code == 503
