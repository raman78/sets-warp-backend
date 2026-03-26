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
#     contributions/YYYY-MM-DD/<uuid>.png   — crop image
#     knowledge.json                        — merged, approved knowledge base
#
# Environment variables (set in Render dashboard):
#   HF_TOKEN        — HF write token (kept SECRET)
#   HF_REPO_ID      — e.g. "sets-sto/warp-knowledge"
#   ADMIN_KEY       — secret key for /admin/merge endpoint
#   MAX_REQ_PER_IP  — rate limit per IP per day (default: 500)
#   GH_TOKEN        — GitHub Personal Access Token (with workflow scope)
#   GH_REPO         — GitHub repository (e.g. "sets-sto/sets-warp-backend")

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

# ── Load .env (for local dev) ─────────────────────────────────────────────────

def _load_env():
    # Look for .env in the current file's directory or its parent
    for candidate in [Path(__file__).parent / '.env', Path(__file__).parent.parent / '.env']:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())
            break

_load_env()

# ── Config from environment ────────────────────────────────────────────────────
HF_TOKEN       = os.environ.get('HF_TOKEN', '')
HF_REPO_ID     = os.environ.get('HF_REPO_ID', 'sets-sto/warp-knowledge')
ADMIN_KEY      = os.environ.get('ADMIN_KEY', '')
MAX_REQ_PER_IP = int(os.environ.get('MAX_REQ_PER_IP', '500'))

# GitHub Config for automated training triggers
GH_TOKEN = os.environ.get('GH_TOKEN', '')
GH_REPO  = os.environ.get('GH_REPO', 'sets-sto/sets-warp-backend')

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
    version='1.1.0',
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
    """Return metadata for the latest centrally-trained model."""
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
    """Return the merged community knowledge base."""
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

    if not _check_rate_limit(client_ip):
        raise HTTPException(429, 'Rate limit exceeded. Try again tomorrow.')

    try:
        png_bytes = base64.b64decode(req.crop_png_b64)
        if not png_bytes.startswith(b'\x89PNG'):
            raise ValueError('not a PNG')
        if len(png_bytes) > 150_000:
            raise ValueError('PNG too large')
    except Exception as e:
        raise HTTPException(400, f'Invalid crop image: {e}')

    if not is_valid_crop(png_bytes):
        raise HTTPException(400, 'Crop rejected: image too uniform or invalid')

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

    today    = date.today().isoformat()
    hf_path  = f'contributions/{today}/{contrib_id}'

    success = _hf_upload_files({
        f'{hf_path}.json':  json.dumps(record, ensure_ascii=False, indent=2).encode('utf-8'),
        f'{hf_path}.png':   png_bytes,
    }, message=f'WARP contribution: {req.item_name}')

    if not success:
        raise HTTPException(503, 'Storage unavailable, please try later')

    _increment_rate_limit(client_ip)
    log.info(f'Contribution accepted: id={contrib_id} item={req.item_name!r}')

    return {'ok': True, 'contribution_id': contrib_id}


@app.post('/webhooks/hf-dataset')
async def hf_dataset_webhook(request: Request):
    """
    Receives HuggingFace Dataset webhook events and triggers GitHub Action training.
    """
    if not GH_TOKEN or not GH_REPO:
        log.debug('HF webhook received but GitHub credentials not configured — skipping trigger')
        return {'ok': True}

    now = time.time()
    last_trigger = getattr(hf_dataset_webhook, '_last_trigger', 0)
    if now - last_trigger < 3600:
        log.debug(f'HF webhook: GitHub trigger skipped (last trigger {int(now - last_trigger)}s ago)')
        return {'ok': True, 'triggered': False, 'reason': 'rate_limited'}
    hf_dataset_webhook._last_trigger = now

    import asyncio
    asyncio.create_task(_trigger_github_workflow())
    return {'ok': True, 'triggered': True}


async def _trigger_github_workflow() -> None:
    """Fire a GitHub Actions workflow dispatch for train_central_model.yml."""
    import urllib.request
    
    url = f'https://api.github.com/repos/{GH_REPO}/actions/workflows/train_central_model.yml/dispatches'
    payload = json.dumps({'ref': 'main'}).encode('utf-8')
    
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            'Authorization': f'token {GH_TOKEN}',
            'Accept':        'application/vnd.github.v3+json',
            'User-Agent':    'WARP-Backend-Trigger',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            log.info(f'GitHub Workflow triggered on {GH_REPO} (Status: {resp.status})')
    except Exception as e:
        log.warning(f'GitHub Workflow trigger failed: {e}')


@app.post('/admin/merge')
async def admin_merge(
    x_admin_key: str = Header(..., alias='X-Admin-Key')
):
    """Admin endpoint: merge confirmed contributions into knowledge.json."""
    if not ADMIN_KEY or x_admin_key != ADMIN_KEY:
        raise HTTPException(403, 'Forbidden')

    contributions = _load_all_contributions_from_hf()
    if not contributions:
        return {'ok': True, 'merged': 0, 'message': 'No contributions found'}

    existing = _load_knowledge_from_hf()
    merged   = dict(existing)

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
        if count >= 2 or (count >= 1 and ph not in merged):
            if merged.get(ph) != winner:
                merged[ph] = winner
                new_entries += 1

    ok = _hf_upload_files({
        'knowledge.json': json.dumps(
            {'knowledge': merged, 'updated_at': datetime.now(timezone.utc).isoformat() + 'Z'},
            ensure_ascii=False, indent=2
        ).encode('utf-8')
    }, message=f'admin_merge: {new_entries} new entries')

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
        std_dev = np.std(img)
        return std_dev >= 10
    except Exception:
        return False


# ── HF Dataset helpers ─────────────────────────────────────────────────────────

def _hf_upload_files(files: dict[str, bytes], message: str = 'WARP auto-upload') -> bool:
    """Upload multiple files to HF Dataset repo atomically."""
    if not HF_TOKEN or not HF_REPO_ID:
        log.error('HF_TOKEN or HF_REPO_ID not set')
        return False
    try:
        from huggingface_hub import HfApi, CommitOperationAdd
        api = HfApi(token=HF_TOKEN)
        operations = [
            CommitOperationAdd(path_in_repo=path, path_or_fileobj=content)
            for path, content in files.items()
        ]
        api.create_commit(
            repo_id=HF_REPO_ID,
            repo_type='dataset',
            operations=operations,
            commit_message=message,
        )
        return True
    except Exception as e:
        log.error(f'HF atomic upload failed: {e}')
        return False


def _load_model_version_from_hf() -> dict:
    """Download models/model_version.json from HF."""
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
    """Download knowledge.json from HF Dataset."""
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
    """List and download all contribution JSON files from HF Dataset (optimized)."""
    if not HF_TOKEN or not HF_REPO_ID:
        return []
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(token=HF_TOKEN)
        elements = api.list_repo_tree(HF_REPO_ID, path_in_repo='contributions', repo_type='dataset', recursive=True)
        json_files = [e.path for e in elements if e.path.endswith('.json')]
        contribs = []
        for f in json_files:
            try:
                local = hf_hub_download(HF_REPO_ID, f, repo_type='dataset', token=HF_TOKEN)
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
    _rate_limit[ip] = {k: v for k, v in _rate_limit[ip].items() if k >= today}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
