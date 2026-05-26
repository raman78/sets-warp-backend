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

import asyncio
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
HF_TOKEN          = os.environ.get('HF_TOKEN', '')
HF_REPO_ID        = os.environ.get('HF_REPO_ID', 'sets-sto/warp-knowledge')
# Icon-dataset target for bulk-crops, screen-types and anchor-grid uploads
# (Phase 1 of backend-proxy migration). Separate from HF_REPO_ID because
# the icon training data lives in its own dataset.
HF_ICONS_REPO_ID  = os.environ.get('HF_ICONS_REPO_ID', 'sets-sto/sto-icon-dataset')
ADMIN_KEY         = os.environ.get('ADMIN_KEY', '')
MAX_REQ_PER_IP    = int(os.environ.get('MAX_REQ_PER_IP', '500'))

# Bulk-endpoint payload limits — guards against accidental floods and HF
# commit-size limits. Per-item caps mirror client-side validation in
# warp/trainer/sync.py.
MAX_BULK_CROPS         = 50
MAX_BULK_SCREEN_TYPES  = 20
MAX_BULK_ANCHOR_GRIDS  = 20
MAX_CROP_PNG_BYTES     = 150_000     # icon crop
MAX_SCREEN_PNG_BYTES   = 2_500_000   # full screenshot
MIN_CROP_PX            = 16
MIN_TEXT_CROP_H        = 10
MIN_TEXT_CROP_W        = 50
MAX_NAME_LEN           = 120
_TEXT_CROP_PREFIXES    = ('ship_type_', 'ship_tier_')
_INSTALL_ID_RE         = re.compile(r'^[a-zA-Z0-9_-]{8,64}$')
_SCREEN_TYPE_RE        = re.compile(r'^[a-zA-Z0-9_-]{1,40}$')

# GitHub Config for automated training triggers
GH_TOKEN = os.environ.get('GH_TOKEN', '')
GH_REPO  = os.environ.get('GH_REPO', 'sets-sto/sets-warp-backend')

# In-memory rate limit: {ip: {date_str: count}}
_rate_limit: dict[str, dict[str, int]] = {}
_rate_limit_lock = asyncio.Lock()

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


# ── Phase 1: bulk-upload request models ───────────────────────────────────────
#
# These mirror the payloads previously built by warp/trainer/sync.py on the
# client side. The client used to call HfApi.create_commit directly with a
# write-scoped HF token; in Phase 2 it will POST these payloads to the
# backend instead, so the token never leaves the server.

class _BulkCropItem(BaseModel):
    slot:         str = Field(..., min_length=1, max_length=80)
    name:         str = Field(..., min_length=1, max_length=MAX_NAME_LEN)
    crop_png_b64: str = Field(..., min_length=100, max_length=200_000)
    ml_name:      str = Field('', max_length=MAX_NAME_LEN)

    @field_validator('name', 'ml_name', 'slot')
    @classmethod
    def _strip_ctrl(cls, v: str) -> str:
        return re.sub(r'[\x00-\x1f\x7f]', '', v).strip()


class BulkCropsRequest(BaseModel):
    install_id: str             = Field(..., min_length=8, max_length=64)
    items:      list[_BulkCropItem] = Field(..., min_length=1, max_length=MAX_BULK_CROPS)


class _ScreenTypeItem(BaseModel):
    png_b64: str = Field(..., min_length=100, max_length=4_000_000)


class ScreenTypesRequest(BaseModel):
    install_id:  str = Field(..., min_length=8, max_length=64)
    screen_type: str = Field(..., min_length=1, max_length=40)
    items:       list[_ScreenTypeItem] = Field(..., min_length=1, max_length=MAX_BULK_SCREEN_TYPES)


class _AnchorGrid(BaseModel):
    # Mirrors the on-disk format produced by warp/trainer/sync.py reading
    # anchors.json: aspect is a float, each slot is a bbox dict with relative
    # coords (x0_rel/y_rel/w_rel/h_rel) plus optional step_rel/count, and
    # for multi-run slots (BOFF abilities split across rows) an extra
    # 'runs' list of sub-bbox dicts. Value type is Any to accommodate that
    # mix; required-key validation happens in the endpoint.
    build_type: str                        = Field(..., min_length=1, max_length=40)
    aspect:     float | None               = Field(None)
    resolution: str                        = Field('', max_length=16)
    slots:      dict[str, dict[str, Any]]  = Field(..., min_length=3)


class AnchorsRequest(BaseModel):
    install_id: str               = Field(..., min_length=8, max_length=64)
    grids:      list[_AnchorGrid] = Field(..., min_length=1, max_length=MAX_BULK_ANCHOR_GRIDS)


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

    if not await _check_and_increment_rate_limit(client_ip):
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

    # Reject poison labels at ingress — virtual classes (__empty__,
    # __inactive__, __boff_*) and leftover dev-test entries must never
    # reach storage. They previously polluted knowledge.json and caused
    # Stage 0 to hard-override real icons to "empty/inactive" at conf=1.0.
    _name = (req.item_name or '').strip()
    if _name.startswith('__') or _name == 'Test Item Name':
        raise HTTPException(400, f'Crop rejected: label {_name!r} not eligible '
                                 f'for community knowledge')

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

    log.info(f'Contribution accepted: id={contrib_id} item={req.item_name!r}')

    return {'ok': True, 'contribution_id': contrib_id}


# ── Phase 1: bulk-upload endpoints ────────────────────────────────────────────
#
# These replace the direct HF writes that warp/trainer/sync.py used to
# perform with a client-side write token. Each endpoint accepts a batch,
# validates per-item, and produces a single HF commit so we stay well
# inside HF API rate limits.

@app.post('/contribute/bulk-crops')
async def contribute_bulk_crops(req: BulkCropsRequest, request: Request):
    """Accept a batch of confirmed crops + annotations.

    Mirrors warp/trainer/sync.py:_upload(): writes PNGs to
    staging/<install_id>/crops/<sha>.png and rewrites
    staging/<install_id>/annotations.jsonl with last-wins dedup per sha.
    All writes happen in a single HF commit.
    """
    client_ip = _get_client_ip(request)
    if not await _check_and_increment_rate_limit(client_ip):
        raise HTTPException(429, 'Rate limit exceeded. Try again tomorrow.')

    install_id = req.install_id.strip()
    if not _INSTALL_ID_RE.match(install_id):
        raise HTTPException(400, 'Invalid install_id format')

    today = date.today().isoformat()
    staging_dir   = f'staging/{install_id}'
    staging_crop  = f'{staging_dir}/crops'
    staging_anno  = f'{staging_dir}/annotations.jsonl'

    files_to_upload: dict[str, bytes] = {}
    new_entries:    list[dict]        = []
    accepted   = 0
    rejected   = 0
    reasons:   list[str] = []

    for item in req.items:
        name = item.name.strip()
        slot = item.slot.strip()
        if not name or not slot:
            rejected += 1
            reasons.append('empty name/slot')
            continue
        if not name.isprintable():
            rejected += 1
            reasons.append('non-printable name')
            continue
        if name.startswith('__') or name == 'Test Item Name':
            rejected += 1
            reasons.append(f'poison label {name!r}')
            continue

        try:
            png_bytes = base64.b64decode(item.crop_png_b64)
        except Exception:
            rejected += 1
            reasons.append('b64 decode failed')
            continue
        if not png_bytes.startswith(b'\x89PNG'):
            rejected += 1
            reasons.append('not a PNG')
            continue
        if len(png_bytes) > MAX_CROP_PNG_BYTES:
            rejected += 1
            reasons.append(f'PNG too large ({len(png_bytes)} B)')
            continue

        err = _check_crop_dims(png_bytes, slot)
        if err:
            rejected += 1
            reasons.append(err)
            continue

        sha = hashlib.sha256(png_bytes).hexdigest()[:32]
        crop_path = f'{staging_crop}/{sha}.png'
        files_to_upload[crop_path] = png_bytes

        entry = {
            'slot':        slot,
            'name':        name,
            'crop_sha256': sha,
            'date':        today,
        }
        if item.ml_name:
            entry['ml_name'] = item.ml_name
        new_entries.append(entry)
        accepted += 1

    if not new_entries:
        raise HTTPException(400, f'All {rejected} items rejected: {reasons[:3]}')

    # Merge annotations.jsonl with last-wins per sha (matches client logic).
    merged_jsonl = _merge_annotations_jsonl(install_id, new_entries)
    files_to_upload[staging_anno] = merged_jsonl

    ok = _hf_upload_files(
        files_to_upload,
        message=f'WARP bulk: {accepted} crops + annotations ({today})',
        repo_id=HF_ICONS_REPO_ID,
    )
    if not ok:
        raise HTTPException(503, 'Storage unavailable, please try later')

    log.info(f'Bulk crops accepted: install={install_id[:8]} accepted={accepted} rejected={rejected}')
    return {'ok': True, 'accepted': accepted, 'rejected': rejected,
            'rejected_reasons': reasons[:10] if rejected else []}


@app.post('/upload/screen-types')
async def upload_screen_types(req: ScreenTypesRequest, request: Request):
    """Accept a batch of screen-type screenshots for one screen_type label."""
    client_ip = _get_client_ip(request)
    if not await _check_and_increment_rate_limit(client_ip):
        raise HTTPException(429, 'Rate limit exceeded. Try again tomorrow.')

    install_id  = req.install_id.strip()
    if not _INSTALL_ID_RE.match(install_id):
        raise HTTPException(400, 'Invalid install_id format')
    stype = req.screen_type.strip()
    if not _SCREEN_TYPE_RE.match(stype):
        raise HTTPException(400, 'Invalid screen_type')

    base_dir = f'staging/{install_id}/screen_types/{stype}'
    files_to_upload: dict[str, bytes] = {}
    accepted = 0
    rejected = 0
    reasons: list[str] = []

    for item in req.items:
        try:
            png_bytes = base64.b64decode(item.png_b64)
        except Exception:
            rejected += 1
            reasons.append('b64 decode failed')
            continue
        if not png_bytes.startswith(b'\x89PNG'):
            rejected += 1
            reasons.append('not a PNG')
            continue
        if len(png_bytes) > MAX_SCREEN_PNG_BYTES:
            rejected += 1
            reasons.append(f'PNG too large ({len(png_bytes)} B)')
            continue

        sha = hashlib.sha256(png_bytes).hexdigest()[:32]
        files_to_upload[f'{base_dir}/{sha}.png'] = png_bytes
        accepted += 1

    if not files_to_upload:
        raise HTTPException(400, f'All {rejected} items rejected: {reasons[:3]}')

    ok = _hf_upload_files(
        files_to_upload,
        message=f'WARP screen types: {accepted} {stype} screenshots',
        repo_id=HF_ICONS_REPO_ID,
    )
    if not ok:
        raise HTTPException(503, 'Storage unavailable, please try later')

    log.info(f'Screen types accepted: install={install_id[:8]} type={stype} accepted={accepted} rejected={rejected}')
    return {'ok': True, 'accepted': accepted, 'rejected': rejected}


@app.post('/upload/anchors')
async def upload_anchors(req: AnchorsRequest, request: Request):
    """Accept a batch of anchor grids (one file per grid, keyed by sha8)."""
    client_ip = _get_client_ip(request)
    if not await _check_and_increment_rate_limit(client_ip):
        raise HTTPException(429, 'Rate limit exceeded. Try again tomorrow.')

    install_id = req.install_id.strip()
    if not _INSTALL_ID_RE.match(install_id):
        raise HTTPException(400, 'Invalid install_id format')

    base_dir = f'staging/{install_id}'
    files_to_upload: dict[str, bytes] = {}
    accepted = 0
    rejected = 0
    reasons: list[str] = []

    for grid in req.grids:
        slots = grid.slots
        if len(slots) < 3:
            rejected += 1
            reasons.append('fewer than 3 slots')
            continue
        # Each bbox must carry the four relative coords; extra keys (step_rel,
        # count) are preserved for downstream consumers.
        _REQUIRED_BBOX_KEYS = ('x0_rel', 'y_rel', 'w_rel', 'h_rel')
        if any(not all(k in v for k in _REQUIRED_BBOX_KEYS) for v in slots.values()):
            rejected += 1
            reasons.append('bbox missing required keys')
            continue

        payload = {
            'build_type': grid.build_type,
            'aspect':     grid.aspect,
            'resolution': grid.resolution,
            'slots':      slots,
        }
        # sort_keys=True must match the client's canonical form so hashes
        # line up and we don't duplicate the same grid as a different file.
        payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        sha8 = hashlib.sha256(payload_json.encode()).hexdigest()[:8]
        files_to_upload[f'{base_dir}/anchors_grid_{sha8}.json'] = payload_json.encode('utf-8')
        accepted += 1

    if not files_to_upload:
        raise HTTPException(400, f'All {rejected} grids rejected: {reasons[:3]}')

    ok = _hf_upload_files(
        files_to_upload,
        message=f'WARP anchors: {accepted} grid entries',
        repo_id=HF_ICONS_REPO_ID,
    )
    if not ok:
        raise HTTPException(503, 'Storage unavailable, please try later')

    log.info(f'Anchors accepted: install={install_id[:8]} accepted={accepted} rejected={rejected}')
    return {'ok': True, 'accepted': accepted, 'rejected': rejected}


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

def _check_crop_dims(png_bytes: bytes, slot: str) -> str | None:
    """Validate crop image dimensions. Returns None if OK, else error message.

    Text crops (ship_type_/ship_tier_) are wide horizontal bands and use
    relaxed height/width minimums instead of the square icon minimum.
    """
    try:
        nparr = np.frombuffer(png_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return 'unreadable image'
        h, w = img.shape[:2]
        if any(slot.startswith(p) for p in _TEXT_CROP_PREFIXES):
            if h < MIN_TEXT_CROP_H or w < MIN_TEXT_CROP_W:
                return f'too small ({w}x{h})'
        else:
            if h < MIN_CROP_PX or w < MIN_CROP_PX:
                return f'too small ({w}x{h})'
        return None
    except Exception as e:
        return str(e)


def _merge_annotations_jsonl(install_id: str, new_entries: list[dict]) -> bytes:
    """Fetch existing staging annotations.jsonl, merge with last-wins per sha.

    Mirrors warp/trainer/sync.py:_append_staging_annotations_to_ops so that
    the central pipeline sees the same per-sha dedup behaviour regardless of
    whether the client uploaded directly (legacy) or via this backend.
    """
    existing_lines = _fetch_staging_annotations(install_id)

    override_shas = {e.get('crop_sha256', '') for e in new_entries if e.get('crop_sha256')}
    kept: list[str] = []
    for line in existing_lines:
        try:
            sha = json.loads(line).get('crop_sha256', '')
        except Exception:
            kept.append(line)
            continue
        if sha and sha in override_shas:
            continue
        kept.append(line)

    # Dedup within new_entries themselves (last wins) so a single batch
    # with duplicate sha doesn't write two conflicting labels.
    seen: dict[str, dict] = {}
    for e in new_entries:
        sha = e.get('crop_sha256', '')
        if sha:
            seen[sha] = e

    combined = kept + [json.dumps(e, ensure_ascii=False) for e in seen.values()]
    return '\n'.join(combined).encode('utf-8')


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

def _hf_upload_files(
    files: dict[str, bytes],
    message: str = 'WARP auto-upload',
    repo_id: str | None = None,
) -> bool:
    """Upload multiple files to HF Dataset repo atomically.

    `repo_id` defaults to HF_REPO_ID (knowledge dataset). Pass
    HF_ICONS_REPO_ID for icon-dataset uploads (bulk crops, screen types,
    anchor grids).
    """
    target = repo_id or HF_REPO_ID
    if not HF_TOKEN or not target:
        log.error('HF_TOKEN or repo_id not set')
        return False
    try:
        from huggingface_hub import HfApi, CommitOperationAdd
        api = HfApi(token=HF_TOKEN)
        operations = [
            CommitOperationAdd(path_in_repo=path, path_or_fileobj=content)
            for path, content in files.items()
        ]
        api.create_commit(
            repo_id=target,
            repo_type='dataset',
            operations=operations,
            commit_message=message,
        )
        return True
    except Exception as e:
        log.error(f'HF atomic upload failed (repo={target}): {e}')
        return False


def _fetch_staging_annotations(install_id: str) -> list[str]:
    """Fetch existing staging/<install_id>/annotations.jsonl lines from HF.

    Returns the raw JSON lines (already stripped of trailing whitespace),
    or [] if the file doesn't exist yet or download fails.
    """
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=HF_ICONS_REPO_ID,
            filename=f'staging/{install_id}/annotations.jsonl',
            repo_type='dataset',
            token=HF_TOKEN or None,
        )
        out: list[str] = []
        with open(local, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(line)
        return out
    except Exception:
        return []


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
        # Take the rightmost IP, added by the trusted Render proxy.
        # A client can forge earlier entries but not the last one.
        return forwarded.split(',')[-1].strip()
    return request.client.host if request.client else 'unknown'


async def _check_and_increment_rate_limit(ip: str) -> bool:
    """Atomically check and increment rate limit. Returns True if allowed."""
    async with _rate_limit_lock:
        today = str(date.today())
        counts = _rate_limit.get(ip, {})
        if counts.get(today, 0) >= MAX_REQ_PER_IP:
            return False
        if ip not in _rate_limit:
            _rate_limit[ip] = {}
        _rate_limit[ip][today] = _rate_limit[ip].get(today, 0) + 1
        _rate_limit[ip] = {k: v for k, v in _rate_limit[ip].items() if k >= today}
        return True


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
