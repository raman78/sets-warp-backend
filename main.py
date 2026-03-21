# sets-warp-backend/main.py
#
# WARP Knowledge Backend — FastAPI service
#
# Deploy to Render / Railway / any VPS.
# The HF_TOKEN (write) lives ONLY here — never in the client app.
#
# Endpoints:
#   POST /contribute   — receive crop PNG + label from WARP clients
#   GET  /knowledge    — serve merged knowledge base (phash → item_name)
#   GET  /health       — liveness check
#
# Storage:
#   Hugging Face Dataset:  <HF_REPO_ID>  (set via env var)
#     contributions/YYYY-MM-DD/<uuid>.json  — raw contributions (pending review)
#     knowledge.json                        — merged, approved knowledge base
#
# Environment variables (set in Render/Railway dashboard):
#   HF_TOKEN        — HF write token (kept SECRET, never in client code)
#   HF_REPO_ID      — e.g. "sets-sto/warp-knowledge"
#   ADMIN_KEY       — secret key for /admin/merge endpoint
#   MAX_REQ_PER_IP  — rate limit per IP per day (default: 500)

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Config from environment ────────────────────────────────────────────────────
HF_TOKEN       = os.environ.get('HF_TOKEN', '')
HF_REPO_ID     = os.environ.get('HF_REPO_ID', 'sets-sto/warp-knowledge')
ADMIN_KEY      = os.environ.get('ADMIN_KEY', '')
MAX_REQ_PER_IP = int(os.environ.get('MAX_REQ_PER_IP', '500'))

# In-memory rate limit: {ip: {date_str: count}}
_rate_limit: dict[str, dict[str, int]] = {}

# In-memory knowledge cache (rebuilt at startup + after each merge)
_knowledge_cache: dict[str, str] = {}
_knowledge_cache_ts: float = 0.0
KNOWLEDGE_CACHE_TTL = 300  # seconds

# In-memory model version cache
_model_version_cache: dict = {}
_model_version_cache_ts: float = 0.0

app = FastAPI(
    title='WARP Knowledge Backend',
    version='1.0.0',
    description='Community knowledge base for SETS-WARP icon recognition',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['GET', 'POST'],
    allow_headers=['*'],
)


# ── Request models ─────────────────────────────────────────────────────────────

class ContributeRequest(BaseModel):
    install_id:   str  = Field(..., min_length=1,  max_length=64)
    phash:        str  = Field(..., pattern=r'^[0-9a-f]{16}$')
    crop_png_b64: str  = Field(..., min_length=100, max_length=200_000)  # ~150KB max
    item_name:    str  = Field(..., min_length=1,  max_length=300)
    wrong_name:   str  = Field('',                 max_length=300)
    confirmed:    bool = True
    warp_version: str  = Field('',                 max_length=20)
    timestamp:    str  = Field('',                 max_length=30)

    @field_validator('item_name', 'wrong_name')
    @classmethod
    def sanitize_name(cls, v: str) -> str:
        # Strip control characters
        return re.sub(r'[\x00-\x1f\x7f]', '', v).strip()

    @field_validator('install_id')
    @classmethod
    def sanitize_install_id(cls, v: str) -> str:
        return re.sub(r'[^a-zA-Z0-9\-_]', '', v)[:64]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get('/health')
async def health():
    return {'status': 'ok', 'repo': HF_REPO_ID}


@app.get('/model/version')
async def get_model_version():
    """
    Return metadata for the latest centrally-trained model.
    Clients compare 'trained_at' with their local model_version.json
    to decide whether to download an update.

    Format: {"available": true, "version": "abc123", "trained_at": "...", ...}
            {"available": false}  — no model published yet
    """
    global _model_version_cache, _model_version_cache_ts

    now = time.time()
    if now - _model_version_cache_ts > KNOWLEDGE_CACHE_TTL:
        _model_version_cache    = _load_model_version_from_hf()
        _model_version_cache_ts = now

    if not _model_version_cache:
        return JSONResponse({'available': False})
    return JSONResponse({'available': True, **_model_version_cache})


@app.get('/knowledge')
async def get_knowledge():
    """
    Return the merged community knowledge base.
    Format: {"knowledge": {"<phash_hex>": "<item_name>", ...}}
    """
    global _knowledge_cache, _knowledge_cache_ts

    now = time.time()
    if now - _knowledge_cache_ts > KNOWLEDGE_CACHE_TTL:
        _knowledge_cache    = _load_knowledge_from_hf()
        _knowledge_cache_ts = now

    return JSONResponse({'knowledge': _knowledge_cache})


@app.post('/contribute')
async def contribute(req: ContributeRequest, request: Request):
    """
    Accept a crop + label contribution from a WARP client.
    Stores raw contribution to HF Dataset contributions/ folder.
    """
    client_ip = _get_client_ip(request)

    # Rate limit
    if not _check_rate_limit(client_ip):
        raise HTTPException(429, 'Rate limit exceeded. Try again tomorrow.')

    # Validate base64 PNG
    try:
        png_bytes = base64.b64decode(req.crop_png_b64)
        if not png_bytes.startswith(b'\x89PNG'):
            raise ValueError('not a PNG')
        if len(png_bytes) > 150_000:
            raise ValueError('PNG too large')
    except Exception as e:
        raise HTTPException(400, f'Invalid crop image: {e}')

    # Validate image content (reject garbage/uniform images)
    if not is_valid_crop(png_bytes):
        raise HTTPException(400, 'Crop rejected: image too uniform or invalid')

    # Build contribution record
    contrib_id  = hashlib.sha256(
        f'{req.install_id}{req.phash}{req.timestamp}'.encode()
    ).hexdigest()[:16]

    record = {
        'contribution_id': contrib_id,
        'install_id':      req.install_id,
        'phash':           req.phash,
        'item_name':       req.item_name,
        'wrong_name':      req.wrong_name,
        'confirmed':       req.confirmed,
        'warp_version':    req.warp_version,
        'timestamp':       req.timestamp or datetime.now(timezone.utc).isoformat() + 'Z',
        'ip_hash':         hashlib.sha256(client_ip.encode()).hexdigest()[:8],
    }

    # Store PNG and metadata to HF
    today    = date.today().isoformat()
    hf_path  = f'contributions/{today}/{contrib_id}'

    success = _hf_upload_files({
        f'{hf_path}.json':  json.dumps(record, ensure_ascii=False, indent=2).encode('utf-8'),
        f'{hf_path}.png':   png_bytes,
    })

    if not success:
        raise HTTPException(503, 'Storage unavailable, please try later')

    _increment_rate_limit(client_ip)
    log.info(f'Contribution accepted: id={contrib_id} item={req.item_name!r}')

    return {'ok': True, 'contribution_id': contrib_id}


@app.post('/webhooks/hf-dataset')
async def hf_dataset_webhook(request: Request):
    """
    Receives HuggingFace Dataset webhook events (push to sto-icon-dataset).
    Triggers a Bitbucket Pipeline run to start automatic model training.

    Configure on HF: Dataset repo → Settings → Webhooks
      URL: https://<your-backend>/webhooks/hf-dataset
      Events: repo.update

    Bitbucket credentials required in environment:
      BB_WORKSPACE     — Bitbucket workspace slug
      BB_REPO_SLUG     — Repository slug (e.g. sets-warp-backend)
      BB_APP_PASSWORD  — Bitbucket App Password with pipeline:write scope
      BB_USERNAME      — Bitbucket username
    """
    BB_WORKSPACE    = os.environ.get('BB_WORKSPACE', '')
    BB_REPO_SLUG    = os.environ.get('BB_REPO_SLUG', 'sets-warp-backend')
    BB_APP_PASSWORD = os.environ.get('BB_APP_PASSWORD', '')
    BB_USERNAME     = os.environ.get('BB_USERNAME', '')

    if not all([BB_WORKSPACE, BB_APP_PASSWORD, BB_USERNAME]):
        # Not configured — silently accept the webhook (don't expose config state)
        log.debug('HF webhook received but BB credentials not configured — skipping trigger')
        return {'ok': True}

    # Rate-limit: at most one pipeline trigger per hour
    now = time.time()
    last_trigger = getattr(hf_dataset_webhook, '_last_trigger', 0)
    if now - last_trigger < 3600:
        log.debug(f'HF webhook: pipeline trigger skipped (last trigger {int(now - last_trigger)}s ago)')
        return {'ok': True, 'triggered': False, 'reason': 'rate_limited'}
    hf_dataset_webhook._last_trigger = now

    # Trigger Bitbucket Pipeline asynchronously (fire-and-forget)
    import asyncio
    asyncio.create_task(_trigger_bb_pipeline(BB_USERNAME, BB_APP_PASSWORD, BB_WORKSPACE, BB_REPO_SLUG))
    return {'ok': True, 'triggered': True}


async def _trigger_bb_pipeline(username: str, password: str, workspace: str, repo: str) -> None:
    """Fire a Bitbucket Pipelines run for the train-central-model custom pipeline."""
    import base64
    import urllib.request

    payload = json.dumps({
        'target': {
            'type':     'pipeline_ref_target',
            'ref_type': 'branch',
            'ref_name': 'main',
            'selector': {
                'type':    'custom',
                'pattern': 'train-central-model',
            },
        }
    }).encode('utf-8')

    credentials = base64.b64encode(f'{username}:{password}'.encode()).decode()
    url = f'https://api.bitbucket.org/2.0/repositories/{workspace}/{repo}/pipelines/'
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            'Authorization': f'Basic {credentials}',
            'Content-Type':  'application/json',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            log.info(f'BB Pipeline triggered: uuid={result.get("uuid")} state={result.get("state", {}).get("name")}')
    except Exception as e:
        log.warning(f'BB Pipeline trigger failed: {e}')


@app.post('/admin/merge')
async def admin_merge(
    x_admin_key: str = Header(..., alias='X-Admin-Key')
):
    """
    Admin endpoint: merge confirmed contributions into knowledge.json.
    Reads all contributions/**, groups by phash, majority-vote on item_name,
    writes merged knowledge.json back to HF.

    Call this after reviewing contributions on HF (weekly cron or manually).
    """
    if not ADMIN_KEY or x_admin_key != ADMIN_KEY:
        raise HTTPException(403, 'Forbidden')

    contributions = _load_all_contributions_from_hf()
    if not contributions:
        return {'ok': True, 'merged': 0, 'message': 'No contributions found'}

    # Existing knowledge as base
    existing = _load_knowledge_from_hf()
    merged   = dict(existing)

    # Group by phash, majority vote
    from collections import Counter
    phash_votes: dict[str, Counter] = {}
    for c in contributions:
        if not c.get('confirmed'):
            continue
        ph   = c.get('phash', '')
        name = c.get('item_name', '').strip()
        if ph and name:
            phash_votes.setdefault(ph, Counter())[name] += 1

    new_entries = 0
    for ph, votes in phash_votes.items():
        winner, count = votes.most_common(1)[0]
        # Require at least 2 confirmations OR 1 if no existing entry
        if count >= 2 or (count >= 1 and ph not in merged):
            if merged.get(ph) != winner:
                merged[ph] = winner
                new_entries += 1

    # Write back to HF
    ok = _hf_upload_files({
        'knowledge.json': json.dumps(
            {'knowledge': merged, 'updated_at': datetime.now(timezone.utc).isoformat() + 'Z'},
            ensure_ascii=False, indent=2
        ).encode('utf-8')
    })

    if ok:
        global _knowledge_cache, _knowledge_cache_ts
        _knowledge_cache    = merged
        _knowledge_cache_ts = time.time()
        log.info(f'Merge complete: {new_entries} new entries, total={len(merged)}')
        return {'ok': True, 'merged': new_entries, 'total': len(merged)}
    else:
        raise HTTPException(503, 'Failed to write knowledge.json to HF')


# ── Validation helpers ─────────────────────────────────────────────────────────

def is_valid_crop(png_bytes: bytes) -> bool:
    """Checks if crop is valid (not garbage/too uniform)."""
    try:
        nparr = np.frombuffer(png_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False

        # Calculate standard deviation - low means uniform color (empty slot)
        std_dev = np.std(img)
        if std_dev < 10: # Threshold for near-uniform images
            return False
        return True
    except Exception:
        return False


# ── HF Dataset helpers ─────────────────────────────────────────────────────────

def _hf_upload_files(files: dict[str, bytes]) -> bool:
    """Upload multiple files to HF Dataset repo. Returns True on success."""
    if not HF_TOKEN or not HF_REPO_ID:
        log.error('HF_TOKEN or HF_REPO_ID not set')
        return False
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        for path, content in files.items():
            api.upload_file(
                path_or_fileobj=content,
                path_in_repo=path,
                repo_id=HF_REPO_ID,
                repo_type='dataset',
                commit_message=f'WARP auto-contribution: {path}',
            )
        return True
    except Exception as e:
        log.error(f'HF upload failed: {e}')
        return False


def _load_model_version_from_hf() -> dict:
    """Download models/model_version.json from HF. Returns {} if not published yet."""
    if not HF_REPO_ID:
        return {}
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename='models/model_version.json',
            repo_type='dataset',
            token=HF_TOKEN or None,
        )
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception as e:
        log.debug(f'models/model_version.json not found: {e}')
        return {}


def _load_knowledge_from_hf() -> dict[str, str]:
    """Download knowledge.json from HF Dataset. Returns {} on failure."""
    if not HF_REPO_ID:
        return {}
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename='knowledge.json',
            repo_type='dataset',
            token=HF_TOKEN or None,
        )
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        return data.get('knowledge', data)
    except Exception as e:
        log.warning(f'knowledge.json load failed: {e}')
        return {}


def _load_all_contributions_from_hf() -> list[dict]:
    """List and download all contribution JSON files from HF Dataset."""
    if not HF_TOKEN or not HF_REPO_ID:
        return []
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api   = HfApi(token=HF_TOKEN)
        files = api.list_repo_files(HF_REPO_ID, repo_type='dataset')
        contribs = []
        for f in files:
            if f.startswith('contributions/') and f.endswith('.json'):
                try:
                    local = hf_hub_download(
                        HF_REPO_ID, f, repo_type='dataset', token=HF_TOKEN
                    )
                    contribs.append(json.loads(Path(local).read_text()))
                except Exception as e:
                    log.debug(f'skip {f}: {e}')
        return contribs
    except Exception as e:
        log.error(f'list contributions failed: {e}')
        return []


# ── Rate limit helpers ─────────────────────────────────────────────────────────

def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get('X-Forwarded-For')
    if forwarded:
        return forwarded.split(',')[0].strip()
    return request.client.host if request.client else 'unknown'


def _check_rate_limit(ip: str) -> bool:
    today = str(date.today())
    counts = _rate_limit.get(ip, {})
    return counts.get(today, 0) < MAX_REQ_PER_IP


def _increment_rate_limit(ip: str) -> None:
    today = str(date.today())
    if ip not in _rate_limit:
        _rate_limit[ip] = {}
    _rate_limit[ip][today] = _rate_limit[ip].get(today, 0) + 1
    # Clean old dates
    _rate_limit[ip] = {k: v for k, v in _rate_limit[ip].items() if k >= today}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
