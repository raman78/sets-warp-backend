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
#   ADMIN_KEY       — legacy /admin/merge gate; endpoint retired (410) per D-G.8
#   MAX_REQ_PER_IP        — rate limit per IP per day (default: 500)
#   MAX_REQ_PER_INSTALL   — rate limit per install_id per day (default: 500)
#   GH_TOKEN        — GitHub Personal Access Token (with workflow scope)
#   GH_REPO         — GitHub repository (e.g. "sets-sto/sets-warp-backend")

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
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
MAX_REQ_PER_IP      = int(os.environ.get('MAX_REQ_PER_IP', '500'))
MAX_REQ_PER_INSTALL = int(os.environ.get('MAX_REQ_PER_INSTALL', '500'))

# Bulk-endpoint payload limits — guards against accidental floods and HF
# commit-size limits. Per-item caps mirror client-side validation in
# warp/trainer/sync.py.
MAX_BULK_CROPS         = 50
MAX_BULK_SCREEN_TYPES  = 20
MAX_BULK_ANCHOR_GRIDS  = 20
MAX_CROP_PNG_BYTES     = 150_000     # icon crop
MAX_SCREEN_PNG_BYTES   = 6_000_000   # full-resolution screenshot (raw bytes)
MIN_CROP_PX            = 16
MIN_TEXT_CROP_H        = 10
MIN_TEXT_CROP_W        = 50
MAX_NAME_LEN           = 120
# A single install's whole SETS-gap ledger arrives in one request. The cap is
# generous because the payload is names only, and an install that legitimately
# hit hundreds of distinct unimportable items is exactly the signal we want.
MAX_SETS_GAP_ITEMS     = 500
# The two causes a gap can have. They are requests to two different projects
# (the wiki vs SETS), so an unknown third value is a client/server version
# mismatch and is rejected rather than silently filed under the wrong one.
SETS_GAP_REASONS       = frozenset({'missing-from-cargo', 'sets-loader-skips'})
_TEXT_CROP_PREFIXES    = ('ship_type_', 'ship_tier_')
_INSTALL_ID_RE         = re.compile(r'^[a-zA-Z0-9_-]{8,64}$')
_SCREEN_TYPE_RE        = re.compile(r'^[a-zA-Z0-9_-]{1,40}$')

# Anchor grid validation (D-G.5). Coord values are relative (0.0-1.0);
# aspect is monitor width/height, from 16:10 portrait (~0.62) up to
# 32:9 ultrawide (~3.56). Resolution, when supplied, must look like
# WIDTHxHEIGHT in pixels.
_ANCHOR_COORD_KEYS = ('x0_rel', 'y_rel', 'w_rel', 'h_rel', 'step_rel')
_ANCHOR_ASPECT_MIN = 0.5
_ANCHOR_ASPECT_MAX = 3.5
_RESOLUTION_RE     = re.compile(r'^\d{3,5}x\d{3,5}$')

# Poison label policy — see docs/data_source_audit.md D-A.1 / D-G.1.
#
# Virtual classes (__empty__, __inactive__, __boff_*) and leftover dev-test
# entries used to be rejected at ingress (every endpoint that wrote into
# staging or contributions). Per D-A.1 they are now legitimate ML labels
# end-to-end; client-side defense-in-depth (sto-warp icon_matcher.py:244)
# still suppresses them as knowledge.json hard-overrides, so the user view
# is protected without rejecting input.
#
# Toggle _POISON_FILTER_ENABLED back to True to restore the previous
# behaviour at every call site simultaneously. Rollback MUST happen in
# lockstep with the client (sto-warp warp/knowledge/sync_client.py) —
# atomic rollback per docs/client_user_view_filter.md Z5-C.3.
_POISON_FILTER_ENABLED = False


def _is_poison_label(name: str) -> bool:
    """Return True if `name` is a virtual class or dev-test placeholder
    that should be rejected at ingress. Always False while the policy
    flag is disabled (D-A.1)."""
    if not _POISON_FILTER_ENABLED:
        return False
    stripped = (name or '').strip()
    return stripped.startswith('__') or stripped == 'Test Item Name'


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

# In-memory whitelist cache (D-G.6). Runtime source of truth is HF
# `<HF_ICONS_REPO_ID>:config/labels.json`; bundled `config/labels.json`
# in the repo is the bootstrap fallback used when HF is unreachable or
# the file has not been seeded yet. Both endpoints `/upload/screen-types`
# and `/upload/anchors` consult this list before touching staging.
_labels_cache: dict = {}
_labels_cache_ts: float = 0.0
LABELS_CACHE_TTL = 300  # seconds

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
    # 8M b64 ≈ 6 MB raw (= MAX_SCREEN_PNG_BYTES × 4/3) — accepts full-resolution
    # screenshots. Keep in sync with sync.py MAX_SCREEN_PNG_B64 and the byte
    # cap enforced in upload_screen_types().
    png_b64: str = Field(..., min_length=100, max_length=8_000_000)


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


class _SetsGapItem(BaseModel):
    # One item WARP recognised that SETS drops on import. No image and no
    # build — the name is the whole payload, because the only question this
    # answers is "how many installs hit this item".
    name:   str       = Field(..., min_length=1, max_length=MAX_NAME_LEN)
    reason: str       = Field(..., min_length=1, max_length=40)
    slots:  list[str] = Field(default_factory=list, max_length=12)


class SetsGapsRequest(BaseModel):
    install_id: str                = Field(..., min_length=8, max_length=64)
    items:      list[_SetsGapItem] = Field(..., max_length=MAX_SETS_GAP_ITEMS)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get('/health')
async def health():
    # `validation` is here because an empty whitelist disables every ingestion
    # gate: without it on the outside, a backend running wide open looks
    # exactly like a healthy one.
    labels = _get_labels()
    n_types = len(labels.get('screen_types') or [])
    return {
        'status': 'ok',
        'repo': HF_REPO_ID,
        'validation': 'enforcing' if n_types else 'DISABLED (empty whitelist)',
        'screen_types': n_types,
    }


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


@app.get('/config/labels')
async def get_config_labels():
    """Return the screen-type / slot whitelist used by the ingestion
    endpoints. Clients MAY use this for pre-flight validation, but the
    backend treats it as the authoritative source — the client copy is
    only a hint.
    """
    return JSONResponse(_get_labels())


@app.get('/quota')
async def get_quota(request: Request, install_id: str = ''):
    """What this caller has spent today, in the unit the caps are counted in.

    A read, so it is not itself rate limited and cannot make the situation it
    reports any worse. It answers two questions a refused client otherwise
    cannot tell apart, because both come back as the same `429`:

    - **Which bucket is full.** A client knows how many requests it sent, but
      not what the server counted, and not the per-IP bucket it shares with
      everyone behind the same address.
    - **Whether the per-IP bucket is per-user at all.** `resolved_ip` is the
      address this server would rate limit the caller under, and
      `forwarded_for` is the header it derived that from. If the resolved
      address is not the caller's own public IP, the bucket is shared by every
      client behind the same infrastructure and `MAX_REQ_PER_IP` is in effect
      a global cap rather than a per-user one — which is the difference
      between "this install is noisy" and "the whole community is locked out
      after 500 requests a day".

    That second question is open because `_get_client_ip` takes the rightmost
    `X-Forwarded-For` entry, which identifies the caller only when exactly one
    trusted proxy sits in front of the app. That was true of the Render
    deployment it was written for; production is now an HF Space, and the
    number of hops there has never been checked.

    Counts are the live buckets, so they include this request's own history
    but not this request — a GET increments nothing.
    """
    ip = _get_client_ip(request)
    today = str(date.today())
    install_key = f'install:{install_id.strip()}' if install_id.strip() else None
    async with _rate_limit_lock:
        ip_used = _rate_limit.get(ip, {}).get(today, 0)
        install_used = (_rate_limit.get(install_key, {}).get(today, 0)
                        if install_key else None)
    return JSONResponse({
        'day':            today,
        'resolved_ip':    ip,
        'forwarded_for':  request.headers.get('X-Forwarded-For', ''),
        'ip':       {'used': ip_used, 'cap': MAX_REQ_PER_IP},
        'install':  ({'id': install_id.strip(), 'used': install_used,
                      'cap': MAX_REQ_PER_INSTALL}
                     if install_key else None),
    })


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

    install_id_key = (req.install_id or '').strip() or None
    if not await _check_and_increment_rate_limit(client_ip, install_id_key):
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

    # Poison-label gate at ingress (D-A.1 / D-G.1). The filter is wired
    # through `_is_poison_label`, which is currently disabled by the
    # `_POISON_FILTER_ENABLED` flag — see policy block at the top of this
    # module for rationale and rollback instructions.
    _name = (req.item_name or '').strip()
    if _is_poison_label(_name):
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
    install_id = req.install_id.strip()
    if not await _check_and_increment_rate_limit(client_ip, install_id or None):
        raise HTTPException(429, 'Rate limit exceeded. Try again tomorrow.')

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
        if _is_poison_label(name):
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

        # D-G.4: image-quality gate (std_dev >= 10) for icon crops only.
        # Text crops (ship_type_/ship_tier_) are wide low-contrast bands —
        # std_dev would reject legitimate captures, so skip them here.
        is_text_crop = any(slot.startswith(p) for p in _TEXT_CROP_PREFIXES)
        if not is_text_crop and not is_valid_crop(png_bytes):
            rejected += 1
            reasons.append('image too uniform')
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
    install_id  = req.install_id.strip()
    if not await _check_and_increment_rate_limit(client_ip, install_id or None):
        raise HTTPException(429, 'Rate limit exceeded. Try again tomorrow.')

    if not _INSTALL_ID_RE.match(install_id):
        raise HTTPException(400, 'Invalid install_id format')
    stype = req.screen_type.strip()
    if not _SCREEN_TYPE_RE.match(stype):
        raise HTTPException(400, 'Invalid screen_type')
    allowed_screen_types = set(_get_labels().get('screen_types') or [])
    if allowed_screen_types and stype not in allowed_screen_types:
        raise HTTPException(400, f'screen_type {stype!r} not in whitelist')

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
    install_id = req.install_id.strip()
    if not await _check_and_increment_rate_limit(client_ip, install_id or None):
        raise HTTPException(429, 'Rate limit exceeded. Try again tomorrow.')

    if not _INSTALL_ID_RE.match(install_id):
        raise HTTPException(400, 'Invalid install_id format')

    base_dir = f'staging/{install_id}'
    files_to_upload: dict[str, bytes] = {}
    accepted = 0
    rejected = 0
    reasons: list[str] = []

    labels         = _get_labels()
    slot_whitelist = _anchor_whitelist(labels)

    for grid in req.grids:
        slots = grid.slots
        if len(slots) < 3:
            rejected += 1
            reasons.append('fewer than 3 slots')
            continue
        bbox_err = next(
            (err for err in (_anchor_bbox_error(v) for v in slots.values())
             if err is not None),
            None,
        )
        if bbox_err is not None:
            rejected += 1
            reasons.append(bbox_err)
            continue

        # D-G.5: aspect range + optional resolution shape.
        if grid.aspect is not None:
            a = grid.aspect
            if not math.isfinite(a) or not (_ANCHOR_ASPECT_MIN <= a <= _ANCHOR_ASPECT_MAX):
                rejected += 1
                reasons.append(f'aspect {a!r} out of range '
                               f'[{_ANCHOR_ASPECT_MIN}, {_ANCHOR_ASPECT_MAX}]')
                continue
        if grid.resolution and not _RESOLUTION_RE.match(grid.resolution):
            rejected += 1
            reasons.append(f'resolution {grid.resolution!r} not WIDTHxHEIGHT')
            continue

        # D-G.10: enforce the build_type + slot-name whitelist sourced from
        # config/labels.json. An empty whitelist (bundled load failed AND HF
        # unreachable) disables enforcement so we don't black-hole production
        # traffic on a transient outage — `_get_labels()` logs the warning.
        if slot_whitelist and grid.build_type not in slot_whitelist:
            rejected += 1
            reasons.append(f'build_type {grid.build_type!r} not in whitelist')
            continue
        # A declared-but-empty slot list means "this build type has no icon
        # slots" — an anchor grid for it is invalid by definition. Only a
        # *missing* entry leaves enforcement off, which is the fail-open above.
        declared_slots = grid.build_type in slot_whitelist
        allowed_slots  = slot_whitelist.get(grid.build_type) or set()
        if declared_slots and not allowed_slots:
            rejected += 1
            reasons.append(f'build_type {grid.build_type!r} has no icon slots')
            continue
        if allowed_slots:
            stray = [k for k in slots.keys()
                     if k not in allowed_slots and not _ANCHOR_SEAT_RE.match(k)]
            if stray:
                rejected += 1
                reasons.append(f'slots not in whitelist for {grid.build_type}: {stray[:3]}')
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


@app.post('/upload/sets-gaps')
async def upload_sets_gaps(req: SetsGapsRequest, request: Request):
    """Accept one install's tally of items SETS drops on import.

    These are items WARP recognised correctly and SETS cannot place — either
    no cargo table stores them (a request to the wiki) or its build loader
    passes the row over (a request to SETS). Collected here so the case for
    fixing either can be made from how many installs actually hit the item,
    rather than from whichever build someone happened to post.

    The request carries the install's **whole current ledger**, and it
    replaces that install's file outright. Two consequences, both wanted:
    re-sending is idempotent, so a retry or a second client on the same
    machine cannot inflate the count; and an item that stops appearing (the
    wiki added the cargo row) drops out on its own, so the list shrinks
    without anyone sweeping it.

    Counting is therefore per install, never per export. One user exporting
    the same build two hundred times is one data point — the opposite would
    make a single enthusiastic user look like demand.

    No image and no build is sent, only item names.
    """
    client_ip  = _get_client_ip(request)
    install_id = req.install_id.strip()
    if not await _check_and_increment_rate_limit(client_ip, install_id or None):
        raise HTTPException(429, 'Rate limit exceeded. Try again tomorrow.')

    if not _INSTALL_ID_RE.match(install_id):
        raise HTTPException(400, 'Invalid install_id format')

    accepted: list[dict] = []
    rejected = 0
    reasons: list[str] = []
    seen: set[tuple[str, str]] = set()

    for item in req.items:
        name = item.name.strip()
        if not name:
            rejected += 1
            reasons.append('empty name')
            continue
        if item.reason not in SETS_GAP_REASONS:
            rejected += 1
            reasons.append(f'unknown reason {item.reason!r}')
            continue
        # The reason is part of the identity: the same name can legitimately
        # appear under both causes, and merging them would collapse two
        # separate upstream requests into one nobody can act on.
        key = (item.reason, name)
        if key in seen:
            continue
        seen.add(key)
        accepted.append({
            'name':   name,
            'reason': item.reason,
            'slots':  sorted({s.strip() for s in item.slots if s.strip()})[:12],
        })

    # An empty ledger is a valid state to report — it is how an install says
    # "I no longer hit any of these". Writing it is what lets stale entries
    # expire instead of being pinned forever by a single old upload.
    payload = json.dumps({
        'install_id': install_id,
        'items':      sorted(accepted, key=lambda e: (e['reason'], e['name'])),
        'updated_at': datetime.now(timezone.utc).isoformat(timespec='seconds')
                                                .replace('+00:00', 'Z'),
    }, ensure_ascii=False, indent=2, sort_keys=True)

    ok = _hf_upload_files(
        {f'sets_gaps/{install_id}.json': payload.encode('utf-8')},
        message=f'WARP SETS gaps: {len(accepted)} item(s) from one install',
        repo_id=HF_ICONS_REPO_ID,
    )
    if not ok:
        raise HTTPException(503, 'Storage unavailable, please try later')

    log.info(f'SETS gaps accepted: install={install_id[:8]} '
             f'items={len(accepted)} rejected={rejected}')
    return {'ok': True, 'accepted': len(accepted), 'rejected': rejected,
            'reasons': reasons[:3]}


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
    x_admin_key: str = Header(None, alias='X-Admin-Key')
):
    """Retired (D-G.8): merge logic now lives in admin_merge.py, scheduled
    every 2 hours by .github/workflows/merge_staging.yml. The HTTP entry
    point would duplicate the CI worker and could race with it on the same
    knowledge.json file. Always returns 410 Gone."""
    raise HTTPException(
        status_code=410,
        detail='/admin/merge retired — democratic merging runs every 2h via '
               'the merge_staging.yml GitHub Action. See admin_merge.py.',
    )


# ── Validation helpers ─────────────────────────────────────────────────────────

def _anchor_coord_value_ok(v: object) -> bool:
    """Coord values must be finite floats in [0.0, 1.0]."""
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    f = float(v)
    return math.isfinite(f) and 0.0 <= f <= 1.0


def _anchor_bbox_error(v: object) -> str | None:
    """Validate one slot bbox. Returns None if OK, else a short reason.

    Required: dict with y_rel/w_rel/h_rel. Either a top-level x0_rel or a
    non-empty `runs` list whose entries carry their own x0_rel. All present
    coord values (incl. optional step_rel) must be finite floats in
    [0.0, 1.0].
    """
    if not isinstance(v, dict):
        return 'bbox not a dict'
    for k in ('y_rel', 'w_rel', 'h_rel'):
        if k not in v:
            return f'bbox missing {k}'
        if not _anchor_coord_value_ok(v[k]):
            return f'bbox {k}={v[k]!r} out of [0.0, 1.0]'
    if 'step_rel' in v and not _anchor_coord_value_ok(v['step_rel']):
        return f'bbox step_rel={v["step_rel"]!r} out of [0.0, 1.0]'
    if 'x0_rel' in v:
        if not _anchor_coord_value_ok(v['x0_rel']):
            return f'bbox x0_rel={v["x0_rel"]!r} out of [0.0, 1.0]'
        return None
    runs = v.get('runs')
    if not isinstance(runs, list) or not runs:
        return 'bbox missing x0_rel and runs'
    for i, r in enumerate(runs):
        if not isinstance(r, dict) or 'x0_rel' not in r:
            return f'runs[{i}] missing x0_rel'
        if not _anchor_coord_value_ok(r['x0_rel']):
            return f'runs[{i}] x0_rel={r["x0_rel"]!r} out of [0.0, 1.0]'
    return None


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
    """
    Download models/model_version.json from HF and stamp the ArcFace embedder
    version onto it.

    The softmax classifier and the ArcFace embedder are published by two
    independent trainers (admin_train.py / admin_train_metric.py) with their own
    cadence, so model_version.json alone cannot tell a client whether a newer
    embedder is waiting. The embedder fields are advisory: when
    icon_embedder_meta.json is missing they are simply absent from the payload.
    """
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
        version = json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception as e:
        log.debug(f'models/model_version.json not found: {e}')
        return {}

    embedder = _load_embedder_meta_from_hf()
    if embedder.get('trained_at'):
        version['embedder_trained_at'] = embedder['trained_at']
        version['embedder_n_classes']  = embedder.get('n_classes', 0)
        version['embedder_recall']     = embedder.get('val_recall@1', 0)
    return version


def _load_embedder_meta_from_hf() -> dict:
    """Download models/icon_embedder_meta.json from HF (empty dict if absent)."""
    if not HF_REPO_ID:
        return {}
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename='models/icon_embedder_meta.json',
            repo_type='dataset',
            token=HF_TOKEN or None,
        )
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception as e:
        log.debug(f'models/icon_embedder_meta.json not found: {e}')
        return {}


def _load_labels_bundled() -> dict:
    """Read the bootstrap labels.json shipped alongside the backend code."""
    path = Path(__file__).parent / 'config' / 'labels.json'
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        # An empty whitelist switches every gate off (see the ingestion
        # endpoints). That fail-open is deliberate for a transient outage, but
        # a missing bundled file is not transient — it silently runs the
        # backend with no validation at all, which is how this shipped for
        # months. Loud, and visible on /health.
        log.error(f'bundled labels.json load failed: {e} — '
                  f'INGESTION VALIDATION IS DISABLED')
        return {'schema_version': 1, 'screen_types': [], 'slots': {}}


def _load_labels_from_hf() -> dict:
    """Download config/labels.json from the HF icons dataset.

    Falls back to the repo-bundled copy if HF is unreachable or the file
    has not been seeded yet. The bundled copy is also returned when
    `HF_ICONS_REPO_ID` is empty (local dev).
    """
    bundled = _load_labels_bundled()
    if not HF_ICONS_REPO_ID:
        return bundled
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_ICONS_REPO_ID,
            filename='config/labels.json',
            repo_type='dataset',
            token=HF_TOKEN or None,
        )
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(data.get('screen_types'), list):
            log.warning('HF labels.json missing screen_types[]; using bundled')
            return bundled
        return data
    except Exception as e:
        log.warning(f'config/labels.json load from HF failed ({e}); using bundled')
        return bundled


# Anchor grids are keyed by the client's *build type*, and that is a different
# vocabulary from the screen types the client learns them on: WARP CORE folds
# SPACE_EQ and SPACE_MIXED into SPACE, GROUND_EQ and GROUND_MIXED into GROUND,
# TRAITS into SPACE_TRAITS (`trainer_window._STYPE_TO_BUILD`). Checking one
# against the other rejected every SPACE and GROUND grid ever sent — 112 of the
# 176 in one maintainer's store — while BOFFS and SPACE_TRAITS went through
# because the two vocabularies happen to spell those the same.
#
# `labels.json` therefore maps each build type to the screen types it is folded
# from, so the slot lists stay declared once, per screen type, and a build type
# inherits the union.
#
# Marker-keyed BOFF seats cannot be enumerated: their names carry what the
# detector saw, down to the marker's Y in pixels (`Boff Seat R[E+O]_484`).
# The shape is fixed, so the shape is what gets checked. Both documented forms
# are covered, including the legacy one with no seat code
# (`warp/recognition/boff_keys.py`).
_ANCHOR_SEAT_RE = re.compile(r'^Boff Seat [LR](\[[A-Za-z+]+\])?_\d+$')


def _anchor_whitelist(labels: dict) -> dict[str, set[str]]:
    """Build type -> the slot names an anchor grid for it may carry.

    An empty result switches enforcement off, the same fail-open the rest of
    the ingestion gates use. A labels.json from before this key existed —
    which is what HF serves until the dataset copy is reseeded — falls back to
    the bundled map rather than disabling the gate, because the bundled copy
    ships with the code that reads it.
    """
    sources = labels.get('anchor_build_types')
    if not isinstance(sources, dict) or not sources:
        sources = _load_labels_bundled().get('anchor_build_types') or {}
        if sources:
            log.info('labels.json has no anchor_build_types; using the bundled map')
    slot_whitelist = labels.get('slots') or {}
    out: dict[str, set[str]] = {}
    for build_type, screen_types in sources.items():
        allowed: set[str] = set()
        for screen_type in screen_types or ():
            allowed |= set(slot_whitelist.get(screen_type) or [])
        out[build_type] = allowed
    return out


def _get_labels() -> dict:
    """Return the cached whitelist, refreshing from HF on TTL expiry."""
    global _labels_cache, _labels_cache_ts
    now = time.time()
    if not _labels_cache or now - _labels_cache_ts > LABELS_CACHE_TTL:
        _labels_cache    = _load_labels_from_hf()
        _labels_cache_ts = now
    return _labels_cache


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


# ── Rate limit helpers ─────────────────────────────────────────────────────────

def _get_client_ip(request: Request) -> str:
    """The address this server rate limits a caller under.

    The rightmost `X-Forwarded-For` entry, on the reasoning that a client can
    forge earlier entries but not the last one — the last is written by the
    proxy the app is actually behind.

    **That reasoning holds for exactly one trusted proxy**, which was the
    Render deployment this was written for. Production is an HF Space and the
    number of hops in front of the app there has not been checked. If it is
    more than one, the rightmost entry is infrastructure — the same value for
    every client — and `MAX_REQ_PER_IP` becomes a global cap on the whole
    community rather than a per-user one.

    `GET /quota` echoes what this returns, alongside the raw header, so the
    question can be settled by one call from a machine with a known public
    address. Do not change the rule here on reasoning alone; take the reading
    first.
    """
    forwarded = request.headers.get('X-Forwarded-For')
    if forwarded:
        return forwarded.split(',')[-1].strip()
    return request.client.host if request.client else 'unknown'


async def _check_and_increment_rate_limit(ip: str, install_id: str | None = None) -> bool:
    """Atomically check and increment rate limit. Returns True if allowed.

    Two independent buckets are enforced:
      - per IP            (cap: MAX_REQ_PER_IP)
      - per install_id    (cap: MAX_REQ_PER_INSTALL) — only when supplied

    Both must be under cap for the request to be admitted; both are
    incremented together so a partial pass cannot leave the buckets desynced.
    """
    async with _rate_limit_lock:
        today = str(date.today())
        if _rate_limit.get(ip, {}).get(today, 0) >= MAX_REQ_PER_IP:
            return False
        install_key = f'install:{install_id}' if install_id else None
        if install_key and _rate_limit.get(install_key, {}).get(today, 0) >= MAX_REQ_PER_INSTALL:
            return False
        for key in filter(None, (ip, install_key)):
            bucket = _rate_limit.setdefault(key, {})
            bucket[today] = bucket.get(today, 0) + 1
            _rate_limit[key] = {k: v for k, v in bucket.items() if k >= today}
        return True


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
