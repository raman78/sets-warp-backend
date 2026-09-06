"""`GET /quota` — what a caller has spent today, and under which address.

A client that is refused sees only `429`. It cannot tell whether its own
install bucket is full or the per-IP one it shares with everyone behind the
same address, and it cannot tell whether that per-IP bucket identifies it at
all. Both questions are answered here, and the endpoint is a read so asking
cannot make the situation worse.

Offline: FastAPI's TestClient, no network and no HF.
"""
from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, '_rate_limit', {})
    return TestClient(main.app)


def _today() -> str:
    return str(date.today())


# ── It reports, it does not spend ─────────────────────────────────────────

def test_reading_the_quota_costs_nothing(client):
    """The caps count requests, so a diagnostic that counted against them
    would be part of the problem it exists to diagnose."""
    for _ in range(5):
        client.get('/quota')
    assert client.get('/quota').json()['ip']['used'] == 0


def test_it_reports_what_has_been_spent(client):
    main._rate_limit['1.2.3.4'] = {_today(): 137}
    body = client.get('/quota',
                      headers={'X-Forwarded-For': '9.9.9.9, 1.2.3.4'}).json()
    assert body['ip']['used'] == 137
    assert body['ip']['cap'] == main.MAX_REQ_PER_IP


def test_yesterdays_spending_is_not_todays(client):
    main._rate_limit['1.2.3.4'] = {'2000-01-01': 500}
    body = client.get('/quota',
                      headers={'X-Forwarded-For': '1.2.3.4'}).json()
    assert body['ip']['used'] == 0


# ── The install bucket ────────────────────────────────────────────────────

def test_the_install_bucket_is_reported_when_asked_for(client):
    main._rate_limit['install:abc123'] = {_today(): 42}
    body = client.get('/quota', params={'install_id': 'abc123'}).json()
    assert body['install'] == {'id': 'abc123', 'used': 42,
                               'cap': main.MAX_REQ_PER_INSTALL}


def test_no_install_id_reports_no_install_bucket(client):
    assert client.get('/quota').json()['install'] is None


def test_the_two_buckets_are_reported_apart(client):
    """Which of them is full is the whole question — a single number would
    not answer it."""
    main._rate_limit['1.2.3.4'] = {_today(): 500}
    main._rate_limit['install:abc123'] = {_today(): 3}
    body = client.get('/quota', params={'install_id': 'abc123'},
                      headers={'X-Forwarded-For': '1.2.3.4'}).json()
    assert body['ip']['used'] == 500
    assert body['install']['used'] == 3


# ── Which address the caller is rate limited under ────────────────────────

def test_it_echoes_the_address_the_server_would_limit_on(client):
    """`_get_client_ip` takes the rightmost forwarded entry, which is the
    caller only when exactly one trusted proxy sits in front of the app. If
    the reported address is not the caller's own, the per-IP cap is shared by
    everyone behind that infrastructure."""
    body = client.get('/quota',
                      headers={'X-Forwarded-For': '203.0.113.7, 10.0.0.1'}).json()
    assert body['resolved_ip'] == '10.0.0.1'


def test_it_echoes_the_header_it_derived_that_from(client):
    """Without the raw chain there is no way to tell a one-hop deployment
    from a two-hop one."""
    chain = '203.0.113.7, 10.0.0.1'
    body = client.get('/quota', headers={'X-Forwarded-For': chain}).json()
    assert body['forwarded_for'] == chain


def test_a_direct_call_falls_back_to_the_peer_address(client):
    body = client.get('/quota').json()
    assert body['resolved_ip']
    assert body['forwarded_for'] == ''
