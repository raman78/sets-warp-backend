#!/usr/bin/env python3
"""
admin_train.py — WARP Central Model Trainer
============================================
Trains two models from community-contributed data:

1. icon_classifier (EfficientNet-B0) — from confirmed icon crops
   staging/<install_id>/crops/<sha>.png  +  annotations.jsonl

2. screen_classifier (MobileNetV3-Small) — from confirmed screen type screenshots
   staging/<install_id>/screen_types/<TYPE>/<sha>.png

Democratic voting: 1 install_id = 1 vote per sha, majority label wins.
Both models uploaded to sets-sto/warp-knowledge/models/.

Requires torch, torchvision, cv2 — installed in the sets-warp venv, not here.
Run from the sets-warp directory:
    .venv/bin/python ../sets-warp-backend/admin_train.py
    .venv/bin/python ../sets-warp-backend/admin_train.py --train --min 1

Environment variables (.env in this directory):
    HF_TOKEN         — HF write token (write access to both repos)
    HF_DATASET       — training crops repo (default: sets-sto/sto-icon-dataset)
    HF_REPO_ID       — model output repo  (default: sets-sto/warp-knowledge)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

UTC = timezone.utc

# ── Load .env ─────────────────────────────────────────────────────────────────

def _load_env():
    for candidate in [Path(__file__).parent / '.env', Path(__file__).parent.parent / '.env']:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())
            break

_load_env()

HF_TOKEN   = os.environ.get('HF_TOKEN', '')
HF_DATASET = os.environ.get('HF_DATASET', 'sets-sto/sto-icon-dataset')
HF_REPO_ID = os.environ.get('HF_REPO_ID', 'sets-sto/warp-knowledge')

# ── Training hyper-parameters (mirror local_trainer.py) ──────────────────────

IMG_SIZE       = 64
MODEL_IMG_SIZE = 224
BATCH_SIZE     = 16
MAX_EPOCHS     = 30
LR             = 3e-4
PATIENCE       = 5
MIN_SAMPLES    = 5   # require at least 5 total crops to bother training
MIN_NEW_CROPS  = 10  # minimum new crops since last training to bother retraining

# Screen classifier hyper-parameters (MobileNetV3-Small)
SC_IMG_SIZE         = 224
SC_BATCH_SIZE       = 8
SC_MAX_EPOCHS       = 40
SC_LR               = 3e-4
SC_PATIENCE         = 8
SC_MIN_SAMPLES      = 7   # at least 7 screenshots total to bother training
SC_MIN_CLASS_SAMPLES = 5  # drop a class from training if it has fewer than this many samples
SC_MIN_KEEP         = 30  # per screen-type: below this count keep all samples
SC_MAX_KEEP         = 150 # per screen-type: above SC_MIN_KEEP cap to this many

# Classes the screen classifier is trained on. Narrower than the ingestion
# whitelist on purpose: SPACE_/GROUND_ variants are stored but not trained as
# separate classes (TRAITS has worked this way from the start), so the model
# stays at one class per visually distinct screen.
SCREEN_TYPES = [
    'SPACE_EQ', 'GROUND_EQ', 'TRAITS',
    'SPACE_BOFFS', 'GROUND_BOFFS', 'BOFFS',
    'SPECIALIZATIONS', 'SPACE_MIXED', 'GROUND_MIXED',
    'SKILLS', 'DISCARD',
]

# ── HF helpers ────────────────────────────────────────────────────────────────

def _require_hf():
    try:
        from huggingface_hub import HfApi, hf_hub_download  # noqa
    except ImportError:
        print('ERROR: pip install huggingface-hub', file=sys.stderr)
        sys.exit(1)
    if not HF_TOKEN:
        print('ERROR: HF_TOKEN not set', file=sys.stderr)
        sys.exit(1)


def _list_staging_folders() -> list[str]:
    """Return list of install_id staging folder names."""
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFolder
    from hf_retry import retry_on_429
    api   = HfApi(token=HF_TOKEN)
    try:
        # Optimization: list only the 'staging/' directory non-recursively
        elements = retry_on_429(
            lambda: list(api.list_repo_tree(HF_DATASET, path_in_repo='staging',
                                            repo_type='dataset', recursive=False)),
            label="list_repo_tree('staging')",
        )
        folders = [e.path.split('/')[-1] for e in elements if isinstance(e, RepoFolder)]
        if folders:
            return sorted(folders)
    except Exception as e:
        log.warning(f"list_repo_tree('staging') failed: {e}. Falling back to full list.")

    # Fallback to the old method (might timeout on large repos)
    files = retry_on_429(
        lambda: list(api.list_repo_files(HF_DATASET, repo_type='dataset')),
        label='list_repo_files(fallback)',
    )
    folders = {
        f.split('/')[1]
        for f in files
        if f.startswith('staging/') and '/' in f[len('staging/'):]
    }
    return sorted(folders)


def _load_staging_annotations(install_id: str) -> list[dict]:
    """Download and parse staging/<install_id>/annotations.jsonl."""
    from huggingface_hub import hf_hub_download
    path_in_repo = f'staging/{install_id}/annotations.jsonl'
    try:
        local = hf_hub_download(
            HF_DATASET, path_in_repo, repo_type='dataset', token=HF_TOKEN
        )
        entries = []
        for line in Path(local).read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
        return entries
    except Exception as e:
        log.debug(f'No annotations for {install_id}: {e}')
        return []


def read_curated_crops() -> tuple[dict[str, str], dict[str, int]]:
    """Read data/annotations.jsonl (written by democratic_merge_crops.py).

    Returns (sha → name, sha → vote_count). The merger has already enforced
    Z3 asymmetric thresholds (NEW=1, UPDATE>=2) and dropped poison labels,
    so the trainer just consumes consensus.
    """
    from huggingface_hub import hf_hub_download

    try:
        local = hf_hub_download(
            repo_id=HF_DATASET, filename='data/annotations.jsonl',
            repo_type='dataset', token=HF_TOKEN,
        )
    except Exception as e:
        log.debug(f'data/annotations.jsonl unavailable: {e}')
        return {}, {}

    labels: dict[str, str] = {}
    votes:  dict[str, int] = {}
    for line in Path(local).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        sha   = (rec.get('crop_sha256') or '').strip()
        name  = (rec.get('name') or '').strip()
        if not (sha and name):
            continue
        labels[sha] = name
        votes[sha]  = int(rec.get('votes') or 1)
    return labels, votes


def _create_commit_with_retry(api, repo_id: str, repo_type: str,
                              operations: list, commit_message: str,
                              max_retries: int = 3) -> bool:
    """Call api.create_commit with retry on 429 rate-limit errors."""
    import re
    from hf_commit import commit_chunked
    for attempt in range(max_retries):
        try:
            # Chunked: one commit for an unbounded operation list is what
            # froze the crop merge for seven weeks (see hf_commit).
            commit_chunked(api, repo_id, repo_type, operations,
                           commit_message)
            return True
        except Exception as e:
            msg = str(e)
            if '429' in msg or 'rate limit' in msg.lower():
                # Parse suggested wait time from HF error ("Retry after N seconds")
                m = re.search(r'[Rr]etry after (\d+)', msg)
                wait = int(m.group(1)) + 10 if m else 150
                if attempt < max_retries - 1:
                    log.warning(f'HF rate limit hit — waiting {wait}s before retry '
                                f'({attempt + 1}/{max_retries - 1})…')
                    time.sleep(wait)
                    continue
            log.error(f'Upload failed: {e}')
            return False
    return False


def _upload_model(models_dir: Path, n_classes: int, val_acc: float,
                  n_samples: int, n_users: int,
                  sc_val_acc: float | None = None,
                  sc_n_samples: int = 0) -> bool:
    """Upload icon + screen model files to sets-sto/warp-knowledge under models/."""
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi(token=HF_TOKEN)

    pt_path     = models_dir / 'icon_classifier.pt'
    label_path  = models_dir / 'label_map.json'
    meta_path   = models_dir / 'icon_classifier_meta.json'

    if not pt_path.exists():
        log.error('icon_classifier.pt not found — nothing to upload')
        return False

    # Compute version hash (sha256 of icon model file, first 16 hex chars)
    sha = hashlib.sha256(pt_path.read_bytes()).hexdigest()[:16]
    trained_at = datetime.now(UTC).isoformat() + 'Z'

    version_data = {
        'version':    sha,
        'trained_at': trained_at,
        'n_classes':  n_classes,
        'val_acc':    round(val_acc, 4),
        'n_samples':  n_samples,
        'n_users':    n_users,
    }
    if sc_val_acc is not None:
        version_data['screen_trained_at'] = trained_at
        version_data['screen_val_acc']    = round(sc_val_acc, 4)
        version_data['screen_n_samples']  = sc_n_samples

    version_path = models_dir / 'model_version.json'
    version_path.write_text(json.dumps(version_data, indent=2), encoding='utf-8')

    manifest_path = models_dir / 'training_manifest.json'
    ops = [
        CommitOperationAdd(path_in_repo='models/icon_classifier.pt',        path_or_fileobj=str(pt_path)),
        CommitOperationAdd(path_in_repo='models/label_map.json',            path_or_fileobj=str(label_path)),
        CommitOperationAdd(path_in_repo='models/icon_classifier_meta.json', path_or_fileobj=str(meta_path)),
        CommitOperationAdd(path_in_repo='models/model_version.json',        path_or_fileobj=str(version_path)),
    ]
    if manifest_path.exists():
        ops.append(CommitOperationAdd(
            path_in_repo='models/training_manifest.json',
            path_or_fileobj=str(manifest_path),
        ))
    # Include screen classifier if trained
    sc_pt     = models_dir / 'screen_classifier.pt'
    sc_labels = models_dir / 'screen_classifier_labels.json'
    if sc_pt.exists():
        ops.append(CommitOperationAdd(path_in_repo='models/screen_classifier.pt',          path_or_fileobj=str(sc_pt)))
    if sc_labels.exists():
        ops.append(CommitOperationAdd(path_in_repo='models/screen_classifier_labels.json', path_or_fileobj=str(sc_labels)))

    commit_msg = (f'admin_train: icon {n_classes}cls val={val_acc:.1%}'
                  + (f', screen val={sc_val_acc:.1%}' if sc_val_acc is not None else '')
                  + f' ({trained_at[:10]})')
    ok = _create_commit_with_retry(api, HF_REPO_ID, 'dataset', ops, commit_msg)
    if ok:
        log.info(f'Model uploaded to {HF_REPO_ID}: version={sha}, val_acc={val_acc:.1%}')
    return ok


# ── Community anchors (P11) ──────────────────────────────────────────────────
#
# PHASE 3 / D-F.2: aggregation lives in democratic_merge_anchors.py now.
# The trainer reads the curated consensus from data/anchors/<build_type>_<bucket>.json
# instead of voting on staging itself. This keeps "one source of truth" — every
# consumer (trainer, future runtime distribution) reads the same files the merger
# produced.

def read_community_anchors() -> list[dict]:
    """Read consensus anchor entries from data/anchors/*.json on HF_DATASET.

    The merger (democratic_merge_anchors.py) writes one JSON per
    (build_type, aspect_bucket) with median-aggregated slot coords +
    spread audit trail. We strip that down to the legacy trainer shape
    `{type, aspect, res, slots, n_contributors, timestamp}` so the
    downstream upload step is unchanged.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=HF_TOKEN)
    try:
        tree = list(api.list_repo_tree(
            HF_DATASET, path_in_repo='data/anchors',
            repo_type='dataset', recursive=False,
        ))
    except Exception as e:
        log.debug(f'data/anchors listing failed: {e}')
        return []

    entries: list[dict] = []
    for item in tree:
        path = getattr(item, 'path', '')
        if not path.endswith('.json'):
            continue
        try:
            local = hf_hub_download(
                repo_id=HF_DATASET, filename=path,
                repo_type='dataset', token=HF_TOKEN,
            )
            body = json.loads(Path(local).read_text(encoding='utf-8'))
        except Exception as e:
            log.debug(f'{path} unavailable: {e}')
            continue
        build_type = (body.get('build_type') or '').strip()
        aspect     = body.get('aspect_bucket')
        slots      = body.get('slots') or {}
        if not (build_type and aspect is not None and slots):
            continue
        entries.append({
            'type':           build_type,
            'aspect':         float(aspect),
            'res':            body.get('representative_resolution', ''),
            'slots':          slots,
            'n_contributors': int(body.get('n_contributors') or 1),
            'timestamp':      int(time.time()),
        })
        print(f'  Loaded community anchor: {build_type} aspect={aspect} '
              f'({entries[-1]["n_contributors"]} contributors, {len(slots)} slots)')

    return entries


def upload_community_anchors(entries: list[dict], models_dir: Path) -> bool:
    """Write community_anchors.json and upload to HF knowledge repo."""
    from huggingface_hub import HfApi, CommitOperationAdd
    from datetime import datetime, timezone
    import io

    payload = {
        'schema_version': 2,
        'generated_at':  datetime.now(timezone.utc).isoformat() + 'Z',
        'n_contributors': max((e['n_contributors'] for e in entries), default=0),
        'entries':        entries,
    }
    payload_bytes = json.dumps(payload, indent=2, ensure_ascii=False).encode('utf-8')

    # Save locally for reference
    local_path = models_dir / 'community_anchors.json'
    local_path.write_bytes(payload_bytes)

    api = HfApi(token=HF_TOKEN)
    ok = _create_commit_with_retry(
        api, HF_REPO_ID, 'dataset',
        [CommitOperationAdd(path_in_repo='models/community_anchors.json',
                            path_or_fileobj=io.BytesIO(payload_bytes))],
        f'community anchors: {len(entries)} entries',
    )
    if ok:
        log.info(f'community_anchors.json uploaded ({len(entries)} entries)')
    return ok


# ── Ship Type / Tier OCR correction map ──────────────────────────────────────
#
# PHASE 3 / D-E.5: voting + tier-poison filtering lives in
# democratic_merge_screens.py now. The trainer reads the curated
# data/text_corrections.jsonl (one consensus winner per ml_name) and
# re-publishes it as models/ship_type_corrections.json on the knowledge
# repo. No staging traversal here anymore.

def publish_text_corrections(models_dir: Path) -> None:
    """Read data/text_corrections.jsonl (consensus already filtered for
    tier-poison upstream) and upload models/ship_type_corrections.json."""
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

    try:
        local = hf_hub_download(
            repo_id=HF_DATASET, filename='data/text_corrections.jsonl',
            repo_type='dataset', token=HF_TOKEN,
        )
    except Exception as e:
        print(f'  data/text_corrections.jsonl unavailable ({e}) — '
              f'ship_type_corrections.json not updated.')
        return

    corrections: dict[str, str] = {}
    for line in Path(local).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        ml_name = (rec.get('ml_name') or '').strip()
        name    = (rec.get('name') or '').strip()
        if ml_name and name and ml_name != name:
            corrections[ml_name] = name

    if not corrections:
        print('  No OCR corrections in data/text_corrections.jsonl — skip upload.')
        return

    print(f'  Loaded {len(corrections)} OCR correction(s) from data/.')

    payload_bytes = json.dumps(corrections, indent=2, ensure_ascii=False).encode('utf-8')
    local_path = models_dir / 'ship_type_corrections.json'
    local_path.write_bytes(payload_bytes)

    api = HfApi(token=HF_TOKEN)
    ok = _create_commit_with_retry(
        api, HF_REPO_ID, 'dataset',
        [CommitOperationAdd(path_in_repo='models/ship_type_corrections.json',
                            path_or_fileobj=local_path)],
        f'ship_type_corrections: {len(corrections)} entries',
    )
    if ok:
        log.info(f'ship_type_corrections.json uploaded ({len(corrections)} entries)')


# ── Screen classifier — read curated data/screen_types/ ───────────────────────

def read_curated_screens() -> tuple[dict[str, str], int]:
    """Return (sha → stype winning label, n_contributors) from data/screen_types/.

    Reads data/screen_types/metadata.jsonl produced by democratic_merge_screens.py.
    Per-class cap (SC_MAX_KEEP) is still applied here — bloat avoidance is a
    training-time choice, not a merge-time one.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=HF_TOKEN)
    try:
        local = hf_hub_download(
            repo_id=HF_DATASET, filename='data/screen_types/metadata.jsonl',
            repo_type='dataset', token=HF_TOKEN,
        )
    except Exception as e:
        log.debug(f'data/screen_types/metadata.jsonl unavailable: {e}')
        return {}, 0

    winner_map: dict[str, str] = {}
    contributor_total = 0
    for line in Path(local).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        sha   = (rec.get('sha') or '').strip()
        stype = (rec.get('type') or '').strip()
        if not (sha and stype and stype in SCREEN_TYPES):
            continue
        winner_map[sha] = stype
        contributor_total = max(contributor_total, int(rec.get('votes') or 0))

    # Per-class cap (mirrors the old behaviour).
    import random as _random
    by_class: dict[str, list[str]] = defaultdict(list)
    for sha, stype in winner_map.items():
        by_class[stype].append(sha)

    capped: dict[str, str] = {}
    for stype, shas in by_class.items():
        if len(shas) >= SC_MIN_KEEP and len(shas) > SC_MAX_KEEP:
            _random.shuffle(shas)
            shas = shas[:SC_MAX_KEEP]
            print(f'  Screen type {stype}: capped to {SC_MAX_KEEP} of '
                  f'{len(by_class[stype])} samples')
        for sha in shas:
            capped[sha] = winner_map[sha]

    return capped, contributor_total


def train_screen_classifier(
    winner_map: dict[str, str],
    models_dir: Path,
    tmpdir: Path,
    prev_model_pt: Path | None = None,
    deadline: float | None = None,
) -> tuple[float, int]:
    """
    Download winning screenshots from data/screen_types/, fine-tune
    MobileNetV3-Small, save to models_dir.

    `winner_map` is now `dict[sha, stype]` — the merger
    (democratic_merge_screens.py) already promoted each sha into
    data/screen_types/<stype>/<sha>.png, so the URL is fully
    determined by (stype, sha). No install_id involved.

    Returns (best_val_acc, n_samples_used).
    """
    import cv2
    import torch
    import torchvision.models as tv_models
    import torchvision.transforms as T
    import torch.nn.functional as _F
    import random
    import logging as _log_sc

    _log_sc.getLogger('httpx').setLevel(_log_sc.WARNING)

    snap_sc = tmpdir / 'snap_sc'
    snap_sc.mkdir(exist_ok=True)

    print(f'\nDownloading {len(winner_map)} screenshots from data/screen_types/...')
    import socket as _socket_sc
    import urllib.request as _urllib_sc
    from concurrent.futures import ThreadPoolExecutor as _TPE_sc

    _socket_sc.setdefaulttimeout(120)
    _hf_base_sc = f'https://huggingface.co/datasets/{HF_DATASET}/resolve/main'
    _opener_sc = _urllib_sc.build_opener()
    if HF_TOKEN:
        _opener_sc.addheaders = [('Authorization', f'Bearer {HF_TOKEN}')]

    def _fetch_screen(args: tuple[str, str]) -> bool:
        sha, stype = args
        dest = snap_sc / 'data' / 'screen_types' / stype / f'{sha}.png'
        if dest.exists():
            return True
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            url = f'{_hf_base_sc}/data/screen_types/{stype}/{sha}.png'
            with _opener_sc.open(url) as r:
                dest.write_bytes(r.read())
            return True
        except Exception:
            return False

    _sc_tasks = list(winner_map.items())
    _sc_ok = _sc_fail = 0
    with _TPE_sc(max_workers=16) as _pool_sc:
        for _r in _pool_sc.map(_fetch_screen, _sc_tasks):
            if _r:
                _sc_ok += 1
            else:
                _sc_fail += 1
    print(f'  {_sc_ok} downloaded, {_sc_fail} failed/skipped.')

    images, labels = [], []
    for sha, stype in winner_map.items():
        p = snap_sc / 'data' / 'screen_types' / stype / f'{sha}.png'
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        images.append(cv2.resize(img, (SC_IMG_SIZE, SC_IMG_SIZE)))
        labels.append(stype)

    print(f'{len(images)}/{len(winner_map)} screenshots loaded.')
    n = len(images)
    print(f'{n} screenshots ready.')
    if n < SC_MIN_SAMPLES:
        raise RuntimeError(
            f'Only {n} screen type screenshots (need {SC_MIN_SAMPLES}). '
            'Contribute more confirmed screen type labels first.'
        )

    # ── Label map ─────────────────────────────────────────────────────────────
    # Log per-class distribution and drop classes with too few samples.
    raw_counts = Counter(labels)
    print('  Per-class sample counts:')
    for lbl in sorted(raw_counts):
        flag = '' if raw_counts[lbl] >= SC_MIN_CLASS_SAMPLES else f'  <-- DROP (< {SC_MIN_CLASS_SAMPLES})'
        print(f'    {lbl:<22}: {raw_counts[lbl]:>4}{flag}')

    kept = {l for l, c in raw_counts.items() if c >= SC_MIN_CLASS_SAMPLES}
    dropped = sorted(raw_counts.keys() - kept)
    if dropped:
        print(f'  Dropping under-represented classes: {dropped}')
        images = [img for img, lbl in zip(images, labels) if lbl in kept]
        labels = [lbl for lbl in labels if lbl in kept]
        n = len(images)
        if n < SC_MIN_SAMPLES:
            raise RuntimeError(f'Only {n} screenshots remain after class filtering (need {SC_MIN_SAMPLES}).')

    unique_labels = sorted(set(labels))
    label_to_idx  = {l: i for i, l in enumerate(unique_labels)}
    idx_to_label  = {i: l for l, i in label_to_idx.items()}
    n_classes     = len(unique_labels)
    y             = [label_to_idx[l] for l in labels]

    print(f'  {n_classes} classes for training: {unique_labels}')

    # ── Dataset ───────────────────────────────────────────────────────────────
    transform_train = T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop(SC_IMG_SIZE, scale=(0.85, 1.0)),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = T.Compose([
        T.ToPILImage(),
        T.Resize((SC_IMG_SIZE, SC_IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class ScreenDataset(torch.utils.data.Dataset):
        def __init__(self, imgs, lbls, tf):
            self.imgs, self.lbls, self.tf = imgs, lbls, tf
        def __len__(self):    return len(self.imgs)
        def __getitem__(self, i):
            return self.tf(cv2.cvtColor(self.imgs[i], cv2.COLOR_BGR2RGB)), self.lbls[i]

    # Stratified split — 20% per class for validation (min 2 if enough samples).
    from collections import defaultdict as _dd_sc
    by_cls_sc: dict[int, list[int]] = _dd_sc(list)
    for i, lbl in enumerate(y):
        by_cls_sc[lbl].append(i)
    train_idx_sc: list[int] = []
    val_idx_sc:   list[int] = []
    for lbl, idxs in by_cls_sc.items():
        random.shuffle(idxs)
        n_val = max(2, len(idxs) // 5) if len(idxs) >= 5 else (1 if len(idxs) >= 2 else 0)
        val_idx_sc.extend(idxs[:n_val])
        train_idx_sc.extend(idxs[n_val:])
    random.shuffle(train_idx_sc)
    val_idx_sc = val_idx_sc or train_idx_sc[:1]

    ds_train = ScreenDataset([images[i] for i in train_idx_sc], [y[i] for i in train_idx_sc], transform_train)
    ds_val   = ScreenDataset([images[i] for i in val_idx_sc],   [y[i] for i in val_idx_sc],   transform_val)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=SC_BATCH_SIZE, shuffle=True,  num_workers=0)
    dl_val   = torch.utils.data.DataLoader(ds_val,   batch_size=SC_BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n── screen_classifier (MobileNetV3-Small) {"─" * 38}')
    print(f'  Dataset : {n} screenshots, {n_classes} classes')
    print(f'  Split   : {len(train_idx_sc)} train / {len(val_idx_sc)} val')
    print(f'  Device  : {device}')
    print(f'{"─" * 64}')

    model = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, n_classes)
    model = model.to(device)

    # Warm-start: load backbone from previous central screen_classifier if available.
    # Strip classifier keys before loading — strict=False ignores missing/unexpected
    # keys but still raises on size mismatch (same key, different n_classes shape).
    sc_fine_tuning = False
    if prev_model_pt and prev_model_pt.exists():
        try:
            state = torch.load(str(prev_model_pt), map_location=device)
            backbone_state = {k: v for k, v in state.items()
                              if not k.startswith('classifier')}
            missing, unexpected = model.load_state_dict(backbone_state, strict=False)
            non_head = [k for k in (missing + unexpected) if 'classifier' not in k]
            if not non_head:
                print('Loaded backbone from previous central screen_classifier — fine-tuning')
                sc_fine_tuning = True
            else:
                print(f'Previous screen model: {len(non_head)} unexpected backbone keys')
        except Exception as e:
            print(f'Previous screen model load failed ({e}) — using ImageNet weights')
    else:
        print('No previous central screen_classifier — training from ImageNet weights')

    if n < 30:
        for p in model.features.parameters():
            p.requires_grad = False

    effective_sc_lr = SC_LR * 0.3 if sc_fine_tuning else SC_LR
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=effective_sc_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SC_MAX_EPOCHS)

    counts = Counter(y)
    _cw = torch.tensor(
        [1.0 / max(counts[i], 1) for i in range(n_classes)],
        dtype=torch.float32, device=device)
    _cw = _cw / _cw.sum() * n_classes
    # Plain cross-entropy with class weights — Focal Loss was miscalibrating
    # softmax outputs (suppressing easy samples → low confidence on correct predictions).
    criterion = torch.nn.CrossEntropyLoss(weight=_cw).to(device)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc   = 0.0
    # Initialise with pre-training weights so we always have a valid fallback,
    # even if val_acc never improves above 0 (e.g. tiny val set or bad warm-start).
    best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_count = 0

    for epoch in range(SC_MAX_EPOCHS):
        if deadline is not None and time.monotonic() > deadline:
            print(f'  Time budget exceeded, stopping screen classifier at epoch {epoch+1}.')
            break
        if epoch == SC_MAX_EPOCHS // 2 and n < 30:
            for p in model.features.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=SC_LR * 0.1)

        model.train()
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                preds   = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total   += yb.size(0)
        val_acc = correct / total if total > 0 else 0.0

        print(f'  Epoch {epoch+1:2d}/{SC_MAX_EPOCHS}  val_acc={val_acc:.1%}  best={best_val_acc:.1%}')

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= SC_PATIENCE:
                print(f'  Early stop at epoch {epoch+1}.')
                break

    if best_state:
        model.load_state_dict(best_state)

    # ── Save ──────────────────────────────────────────────────────────────────
    models_dir.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    torch.save(model.state_dict(), str(models_dir / 'screen_classifier.pt'))
    with open(models_dir / 'screen_classifier_labels.json', 'w', encoding='utf-8') as f:
        json.dump(idx_to_label, f, ensure_ascii=False, indent=2)

    print(f'\n✓ screen_classifier saved — {n_classes} classes, val_acc={best_val_acc:.1%}')
    return best_val_acc, n


# ── Training ──────────────────────────────────────────────────────────────────
#
# PHASE 3 / D-C.1: voting + label deduplication lives in
# democratic_merge_crops.py. The trainer reads the curated
# data/annotations.jsonl + data/crops/<sha>.png artefact (Z1: one source
# of truth) and trains. No staging traversal here anymore.

def train(winner_labels: dict[str, str],
          models_dir: Path, tmpdir: Path,
          prev_model_pt: Path | None = None,
          deadline: float | None = None) -> tuple[float, int]:
    """
    Download winning crops from data/crops/, train EfficientNet-B0, save
    model to models_dir.

    prev_model_pt: path to a previously-trained icon_classifier.pt — its
    backbone weights are loaded (strict=False) for warm-start fine-tuning.

    Returns (best_val_acc, n_samples_used).
    """
    import cv2
    import torch
    import torchvision.models as tv_models
    import torchvision.transforms as T
    import torch.nn.functional as _F
    import random
    from collections import Counter as _Counter

    # ── Collect crops (parallel urllib download with socket timeout) ─────────
    # urllib.request uses blocking sockets → socket.setdefaulttimeout applies,
    # killing stalled TCP reads that httpx/snapshot_download cannot time out.
    import socket as _socket
    import urllib.request as _urllib
    from concurrent.futures import ThreadPoolExecutor as _TPE

    snap_cache = tmpdir / 'snap'
    snap_cache.mkdir(exist_ok=True)

    _socket.setdefaulttimeout(120)  # 2 min hard timeout per socket read
    _hf_base = f'https://huggingface.co/datasets/{HF_DATASET}/resolve/main'
    _auth_headers = [('Authorization', f'Bearer {HF_TOKEN}')] if HF_TOKEN else []
    _opener = _urllib.build_opener()
    _opener.addheaders = _auth_headers

    def _fetch_crop(sha: str) -> bool:
        dest = snap_cache / 'data' / 'crops' / f'{sha}.png'
        if dest.exists():
            return True
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with _opener.open(f'{_hf_base}/data/crops/{sha}.png') as r:
                dest.write_bytes(r.read())
            return True
        except Exception:
            return False

    all_shas = list(winner_labels.keys())
    print(f'\nDownloading {len(all_shas)} crops from data/crops/...')
    _ok = _fail = 0
    with _TPE(max_workers=16) as _pool:
        for _result in _pool.map(_fetch_crop, all_shas):
            if _result:
                _ok += 1
            else:
                _fail += 1
    print(f'  {_ok} downloaded, {_fail} failed/skipped.')

    crops, labels = [], []
    crop_dir = snap_cache / 'data' / 'crops'
    for sha, label in winner_labels.items():
        p = crop_dir / f'{sha}.png'
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        crops.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        labels.append(label)

    print(f'{len(crops)}/{len(winner_labels)} crops loaded.')
    n = len(crops)
    print(f'{n} crops ready.')
    if n < MIN_SAMPLES:
        raise RuntimeError(
            f'Only {n} crops available (need {MIN_SAMPLES}). '
            'Contribute more confirmed annotations first.'
        )

    # ── Label map ────────────────────────────────────────────────────────────
    unique_labels = sorted(set(labels))
    label_to_idx  = {l: i for i, l in enumerate(unique_labels)}
    idx_to_label  = {i: l for l, i in label_to_idx.items()}
    n_classes     = len(unique_labels)
    y             = [label_to_idx[l] for l in labels]

    print(f'{n_classes} classes: {unique_labels[:10]}{"..." if n_classes > 10 else ""}')

    # ── Dataset ──────────────────────────────────────────────────────────────
    transform_train = T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop(MODEL_IMG_SIZE, scale=(0.8, 1.0)),
        # P7: augmentation — reduces overfitting on community crop datasets
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.RandomHorizontalFlip(p=0.3),
        T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = T.Compose([
        T.ToPILImage(),
        T.Resize((MODEL_IMG_SIZE, MODEL_IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class CropDataset(torch.utils.data.Dataset):
        def __init__(self, crops, labels, tf):
            self.crops, self.labels, self.tf = crops, labels, tf
        def __len__(self):    return len(self.crops)
        def __getitem__(self, i):
            return self.tf(cv2.cvtColor(self.crops[i], cv2.COLOR_BGR2RGB)), self.labels[i]

    # Stratified split — classes with 1 sample stay in train only
    from collections import defaultdict as _dd2
    by_cls: dict[int, list[int]] = _dd2(list)
    for i, lbl in enumerate(y):
        by_cls[lbl].append(i)
    train_idx: list[int] = []
    val_idx:   list[int] = []
    for lbl, idxs in by_cls.items():
        random.shuffle(idxs)
        if len(idxs) >= 2:
            val_idx.append(idxs[0])
            train_idx.extend(idxs[1:])
        else:
            train_idx.extend(idxs)
    random.shuffle(train_idx)
    val_idx = val_idx or train_idx[:1]

    ds_train = CropDataset([crops[i] for i in train_idx], [y[i] for i in train_idx], transform_train)
    ds_val   = CropDataset([crops[i] for i in val_idx],   [y[i] for i in val_idx],   transform_val)

    # P9: hard negatives mining — weights updated per epoch
    sample_weights = torch.ones(len(ds_train), dtype=torch.float32)
    sampler  = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(ds_train), replacement=True)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    dl_val   = torch.utils.data.DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n── icon_classifier (EfficientNet-B0) {"─" * 41}')
    print(f'  Dataset : {n} crops, {n_classes} classes')
    print(f'  Device  : {device}')
    print(f'{"─" * 64}')

    model = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, n_classes)
    model = model.to(device)

    # Warm-start: load backbone from previous central model if available.
    # Strip classifier keys before loading — strict=False ignores missing/unexpected
    # keys but still raises on size mismatch (same key, different n_classes shape).
    fine_tuning = False
    if prev_model_pt and prev_model_pt.exists():
        try:
            state = torch.load(str(prev_model_pt), map_location=device)
            backbone_state = {k: v for k, v in state.items()
                              if not k.startswith('classifier')}
            missing, unexpected = model.load_state_dict(backbone_state, strict=False)
            non_head = [k for k in (missing + unexpected) if 'classifier' not in k]
            if not non_head:
                print('Loaded backbone from previous central model — fine-tuning')
                fine_tuning = True
            else:
                print(f'Previous model: {len(non_head)} unexpected backbone keys')
        except Exception as e:
            print(f'Previous model load failed ({e}) — using ImageNet weights')
    else:
        print('No previous central model — training from ImageNet weights')

    if n < 50:
        for p in model.features.parameters():
            p.requires_grad = False

    effective_lr = LR * 0.3 if fine_tuning else LR
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=effective_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    counts = _Counter(y)
    _cw = torch.tensor(
        [1.0 / max(counts[i], 1) for i in range(n_classes)],
        dtype=torch.float32, device=device)
    _cw = _cw / _cw.sum() * n_classes

    # Plain cross-entropy with class weights — Focal Loss was miscalibrating
    # softmax outputs on this small balanced dataset (low confidence on correct predictions).
    criterion = torch.nn.CrossEntropyLoss(weight=_cw).to(device)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_acc   = 0.0
    best_state     = None
    patience_count = 0

    for epoch in range(MAX_EPOCHS):
        if deadline is not None and time.monotonic() > deadline:
            print(f'  Time budget exceeded, stopping icon classifier at epoch {epoch+1}.')
            break
        if epoch == MAX_EPOCHS // 2 and n < 50:
            for p in model.features.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR * 0.1)

        model.train()
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                preds   = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total   += yb.size(0)
        val_acc = correct / total if total > 0 else 0.0

        print(f'  Epoch {epoch+1:2d}/{MAX_EPOCHS}  val_acc={val_acc:.1%}  best={best_val_acc:.1%}')

        # P9: hard negatives — re-weight samples the model got wrong with high confidence
        model.eval()
        with torch.no_grad():
            all_logits, all_targets = [], []
            for xb, yb in torch.utils.data.DataLoader(
                    ds_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=0):
                all_logits.append(model(xb.to(device)).cpu())
                all_targets.append(yb)
            logits_all  = torch.cat(all_logits)
            targets_all = torch.cat(all_targets)
            probs_all   = torch.softmax(logits_all, dim=1)
            pred_all    = logits_all.argmax(dim=1)
            conf_all    = probs_all.gather(1, pred_all.unsqueeze(1)).squeeze(1)
            wrong_mask  = (pred_all != targets_all) & (conf_all > 0.5)
            sample_weights = torch.clamp(sample_weights + wrong_mask.float(), max=3.0)
            sampler.weights.copy_(sample_weights)

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f'  Early stop at epoch {epoch+1}.')
                break

    if best_state:
        model.load_state_dict(best_state)

    # ── Save ─────────────────────────────────────────────────────────────────
    models_dir.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    torch.save(model.state_dict(), str(models_dir / 'icon_classifier.pt'))
    with open(models_dir / 'label_map.json', 'w', encoding='utf-8') as f:
        json.dump(idx_to_label, f, ensure_ascii=False, indent=2)
    with open(models_dir / 'icon_classifier_meta.json', 'w', encoding='utf-8') as f:
        json.dump({'n_classes': n_classes, 'input_size': MODEL_IMG_SIZE}, f)

    print(f'\n✓ icon_classifier saved — {n_classes} classes, val_acc={best_val_acc:.1%}')
    return best_val_acc, n


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_training_manifest() -> set[str]:
    """Download models/training_manifest.json from HF. Returns set of crop SHAs used last time."""
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            HF_REPO_ID, 'models/training_manifest.json',
            repo_type='dataset', token=HF_TOKEN or None,
        )
        data = json.loads(Path(local).read_text(encoding='utf-8'))
        return set(data.get('crop_shas', []))
    except Exception:
        return set()


def _save_training_manifest(crop_shas: set[str], models_dir: Path) -> None:
    """Save training manifest (set of crop SHAs) to models_dir for upload."""
    manifest = {
        'crop_shas':  sorted(crop_shas),
        'updated_at': datetime.now(UTC).isoformat() + 'Z',
        'count':      len(crop_shas),
    }
    (models_dir / 'training_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8'
    )


def main():
    parser = argparse.ArgumentParser(
        description='WARP Central Model Trainer — democratic voting + EfficientNet training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python admin_train.py                    # dry-run: count crops, show vote summary
  python admin_train.py --train            # train and upload model to HF
  python admin_train.py --train --min 2    # require 2 users to agree on a label
  python admin_train.py --train --skip-if-unchanged  # no-op when no new crops

Automated (Bitbucket Pipelines / cron):
  python admin_train.py --train --skip-if-unchanged --min 1

Environment (.env or env vars in CI):
  HF_TOKEN   — HF write token (required)
  HF_DATASET — training crops repo (default: sets-sto/sto-icon-dataset)
  HF_REPO_ID — model output repo  (default: sets-sto/warp-knowledge)
""",
    )
    parser.add_argument('--train', action='store_true',
                        help='Train and upload model (default: dry-run)')
    parser.add_argument('--min',   type=int, default=1, metavar='N',
                        help='Minimum unique users per crop label (default: 1)')
    parser.add_argument('--skip-if-unchanged', action='store_true',
                        help='Skip training when staging crops match the last training manifest')
    parser.add_argument('--force', action='store_true',
                        help='Force re-training even when crops are unchanged (overrides --skip-if-unchanged)')
    args = parser.parse_args()

    _require_hf()

    print('=' * 60)
    print('WARP Central Model Trainer')
    print(f'Dataset: {HF_DATASET}')
    print(f'Output:  {HF_REPO_ID}')
    print(f'Mode:    {"TRAIN + UPLOAD" if args.train else "DRY-RUN"}')
    print(f'Min votes: {args.min}')
    if args.skip_if_unchanged:
        print('Skip-if-unchanged: ON')
    if args.force:
        print('Force retrain: ON')
    print('=' * 60)

    # 1. Read curated consensus from data/annotations.jsonl (Z1: one source of
    # truth — the merger has already enforced Z3 thresholds and dropped poison).
    print('\nReading curated crop labels from data/annotations.jsonl...')
    winner_labels, vote_counts = read_curated_crops()

    if not winner_labels:
        print('No curated annotations found — nothing to do.')
        return

    print(f'  {len(winner_labels)} crops, '
          f'avg votes/crop={sum(vote_counts.values())/max(len(vote_counts),1):.1f}')

    # 2b. Skip-if-unchanged / MIN_NEW_CROPS check (fast path before downloading)
    if args.skip_if_unchanged and args.train and not args.force:
        current_shas = set(winner_labels.keys())
        last_shas    = _load_training_manifest()
        if last_shas and current_shas == last_shas:
            print(f'\nNo new crops since last training ({len(current_shas)} crops unchanged) — skipping.')
            return
        new_count = len(current_shas - last_shas)
        if last_shas and new_count < MIN_NEW_CROPS:
            print(f'\nOnly {new_count} new crop(s) (threshold: {MIN_NEW_CROPS}) — skipping.')
            return
        print(f'{new_count} new crop(s) since last training — proceeding.')

    # Apply min-votes filter — re-read from data/ since vote counts are stamped
    # on each consensus entry (no staging traversal needed).
    if args.min > 1:
        print(f'\nApplying min-votes={args.min} filter...')
        winner_labels = {
            sha: label for sha, label in winner_labels.items()
            if vote_counts.get(sha, 1) >= args.min
        }
        print(f'{len(winner_labels)} crops pass the {args.min}-vote threshold.')

    # Count classes
    from collections import Counter
    label_counts = Counter(winner_labels.values())
    print(f'\n{len(winner_labels)} total crops, {len(label_counts)} classes')
    print('Top 10 classes by crop count:')
    for label, cnt in label_counts.most_common(10):
        print(f'  {cnt:4d}  {label}')

    if not args.train:
        print('\nDRY-RUN complete — use --train to train and upload model.')
        return

    # 3. Train
    try:
        import torch
    except ImportError:
        print('\nERROR: PyTorch not available. Run with sets-warp .venv.', file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir     = Path(tmp)
        models_dir = tmpdir / 'models'

        # Download previous central models for warm-start fine-tuning
        from huggingface_hub import hf_hub_download as _hf_dl
        import shutil as _shutil

        prev_icon_pt = tmpdir / 'prev_icon_classifier.pt'
        try:
            _local = _hf_dl(HF_REPO_ID, 'models/icon_classifier.pt',
                             repo_type='dataset', token=HF_TOKEN or None)
            _shutil.copy2(_local, prev_icon_pt)
            print('Previous icon_classifier.pt downloaded for fine-tuning.')
        except Exception as _e:
            print(f'No previous icon_classifier.pt ({_e}) — will train from ImageNet.')
            prev_icon_pt = None

        prev_sc_pt = tmpdir / 'prev_screen_classifier.pt'
        try:
            _local = _hf_dl(HF_REPO_ID, 'models/screen_classifier.pt',
                             repo_type='dataset', token=HF_TOKEN or None)
            _shutil.copy2(_local, prev_sc_pt)
            print('Previous screen_classifier.pt downloaded for fine-tuning.')
        except Exception as _e:
            print(f'No previous screen_classifier.pt ({_e}) — will train from ImageNet.')
            prev_sc_pt = None

        # Allow 50 min for training (leaves ~10 min buffer for upload within 60 min CI timeout)
        _train_deadline = time.monotonic() + 50 * 60

        print('\nTraining EfficientNet-B0 (icon classifier)...')
        val_acc, n_samples = train(winner_labels, models_dir, tmpdir,
                                   prev_model_pt=prev_icon_pt, deadline=_train_deadline)

        # Train screen_classifier if data available
        sc_val_acc: float | None = None
        sc_n_samples = 0
        print('\nReading curated screen-type consensus from data/screen_types/...')
        try:
            sc_winner_map, sc_max_votes = read_curated_screens()
            sc_counts = Counter(sc_winner_map.values())
            print(f'{len(sc_winner_map)} unique screenshots, {len(sc_counts)} classes: '
                  + ', '.join(f'{k}={v}' for k, v in sorted(sc_counts.items())))
            if len(sc_winner_map) >= SC_MIN_SAMPLES:
                print(f'\nTraining MobileNetV3-Small (screen classifier, peak {sc_max_votes} vote(s) per sha)...')
                # Separate 8-min deadline — screen classifier is fast (lightweight model,
                # small dataset) and must not share the icon classifier's exhausted budget.
                _sc_deadline = time.monotonic() + 8 * 60
                sc_val_acc, sc_n_samples = train_screen_classifier(
                    sc_winner_map, models_dir, tmpdir, prev_model_pt=prev_sc_pt,
                    deadline=_sc_deadline)
            else:
                print(f'Not enough screen type data ({len(sc_winner_map)} < {SC_MIN_SAMPLES}) — skipping screen classifier training.')
        except Exception as e:
            print(f'WARNING: screen classifier training failed: {e}', file=sys.stderr)

        # 4. Upload
        # Save training manifest (so next run can skip if nothing changed)
        _save_training_manifest(set(winner_labels.keys()), models_dir)

        # `n_users` is now reported as the peak vote count across all consensus
        # crops — a lower-bound proxy for unique contributors, since the curated
        # artefact no longer carries per-install attribution.
        n_users_proxy = max(vote_counts.values(), default=0)
        print('\nUploading models to HF...')
        ok = _upload_model(models_dir, len(label_counts), val_acc, n_samples, n_users_proxy,
                           sc_val_acc=sc_val_acc, sc_n_samples=sc_n_samples)
        if ok:
            print(f'\nDone — models published to {HF_REPO_ID}/models/')
        else:
            print('\nERROR — upload failed.', file=sys.stderr)
            sys.exit(1)

        # 5. Community anchors (P11) — read consensus from data/anchors/ (written by
        # democratic_merge_anchors.py) and re-publish to the knowledge repo as
        # community_anchors.json. No staging traversal here anymore.
        print('\nReading community anchors from data/anchors/...')
        anchor_entries = read_community_anchors()
        if anchor_entries:
            upload_community_anchors(anchor_entries, models_dir)
        else:
            print('No community anchors to upload yet.')

        # 6. Ship Type / Tier OCR corrections — read curated consensus from
        # data/text_corrections.jsonl and re-publish to the knowledge repo.
        print('\nPublishing ship type OCR correction map from data/...')
        publish_text_corrections(models_dir)


if __name__ == '__main__':
    main()
