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
FOCAL_GAMMA    = 2.0
MIN_SAMPLES    = 5   # require at least 5 total crops to bother training
MIN_NEW_CROPS  = 10  # minimum new crops since last training to bother retraining

# Screen classifier hyper-parameters (MobileNetV3-Small)
SC_IMG_SIZE    = 224
SC_BATCH_SIZE  = 8
SC_MAX_EPOCHS  = 40
SC_LR          = 3e-4
SC_PATIENCE    = 8
SC_MIN_SAMPLES = 7   # at least 7 screenshots total to bother training
SC_MIN_KEEP    = 30  # per screen-type: below this count keep all samples
SC_MAX_KEEP    = 150 # per screen-type: above SC_MIN_KEEP cap to this many

SCREEN_TYPES = [
    'SPACE_EQ', 'GROUND_EQ', 'TRAITS',
    'BOFFS', 'SPECIALIZATIONS', 'SPACE_MIXED', 'GROUND_MIXED',
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
    api   = HfApi(token=HF_TOKEN)
    try:
        # Optimization: list only the 'staging/' directory non-recursively
        elements = api.list_repo_tree(HF_DATASET, path_in_repo='staging', repo_type='dataset', recursive=False)
        folders = [e.path.split('/')[-1] for e in elements if isinstance(e, RepoFolder)]
        if folders:
            return sorted(folders)
    except Exception as e:
        log.warning(f"list_repo_tree('staging') failed: {e}. Falling back to full list.")

    # Fallback to the old method (might timeout on large repos)
    files = list(api.list_repo_files(HF_DATASET, repo_type='dataset'))
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


def _download_crop(install_id: str, sha: str, dest_dir: Path) -> Path | None:
    """Download staging/<install_id>/crops/<sha>.png to dest_dir. Returns local path."""
    from huggingface_hub import hf_hub_download
    dest = dest_dir / sha
    if dest.exists():
        return dest
    try:
        local = hf_hub_download(
            HF_DATASET,
            f'staging/{install_id}/crops/{sha}.png',
            repo_type='dataset',
            token=HF_TOKEN,
        )
        import shutil
        shutil.copy2(local, dest)
        return dest
    except Exception as e:
        log.debug(f'Crop {sha} from {install_id} missing: {e}')
        return None


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

    msg = (f'admin_train: icon {n_classes}cls val={val_acc:.1%} ({n_samples}s/{n_users}u)'
           + (f'; screen {sc_n_samples}s val={sc_val_acc:.1%}'
              if sc_val_acc is not None else ''))
    try:
        api.create_commit(
            repo_id=HF_REPO_ID,
            repo_type='dataset',
            operations=ops,
            commit_message=f'admin_train: icon {n_classes}cls val={val_acc:.1%}'
                           + (f', screen val={sc_val_acc:.1%}' if sc_val_acc is not None else '')
                           + f' ({trained_at[:10]})',
        )
        log.info(f'Model uploaded to {HF_REPO_ID}: version={sha}, val_acc={val_acc:.1%}')
        return True
    except Exception as e:
        log.error(f'Upload failed: {e}')
        return False


# ── Community anchors (P11) ──────────────────────────────────────────────────

def _list_anchor_grid_files(folders: list[str]) -> list[tuple[str, str]]:
    """Return list of (install_id, filename) for all anchors_grid_*.json in staging."""
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    result = []
    for iid in folders:
        try:
            tree = api.list_repo_tree(
                HF_DATASET, path_in_repo=f'staging/{iid}',
                repo_type='dataset', recursive=False,
            )
            for item in tree:
                p = getattr(item, 'path', '')
                fname = p.split('/')[-1]
                if fname.startswith('anchors_grid_') and fname.endswith('.json'):
                    result.append((iid, fname))
        except Exception:
            pass
    return result


def _download_anchor_grid(install_id: str, filename: str) -> dict | None:
    """Download staging/<install_id>/<filename> and return parsed dict."""
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=HF_DATASET,
            filename=f'staging/{install_id}/{filename}',
            repo_type='dataset',
            token=HF_TOKEN,
        )
        return json.loads(Path(local).read_text(encoding='utf-8'))
    except Exception as e:
        log.debug(f'anchor grid {install_id}/{filename} unavailable: {e}')
        return None


def build_community_anchors(folders: list[str], min_contributors: int = 3) -> list[dict]:
    """
    Aggregate anchor grids from all staging folders.
    Returns list of community anchor entries (same format as anchors.json learned entries).
    Accepts groups with exactly 1 contributor (no conflict) or >= min_contributors (consensus).
    Skips groups with 2..min_contributors-1 (ambiguous — some data but not enough for consensus).
    """
    from collections import defaultdict
    import statistics

    grid_files = _list_anchor_grid_files(folders)
    print(f'Found {len(grid_files)} anchor grid file(s) across {len(folders)} user(s).')

    if not grid_files:
        return []

    # {(build_type, aspect_bucket): {install_id: [grid_entry, ...]}}
    groups: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for install_id, filename in grid_files:
        entry = _download_anchor_grid(install_id, filename)
        if not entry:
            continue
        build_type = entry.get('build_type', '')
        aspect     = entry.get('aspect')
        if not build_type or aspect is None:
            continue
        # Bucket aspect to 2 decimal places for grouping
        aspect_bucket = round(float(aspect), 2)
        groups[(build_type, aspect_bucket)][install_id].append(entry)

    results = []
    for (build_type, aspect_bucket), contributors in groups.items():
        n = len(contributors)
        # Accept sole contributor (no conflict) or consensus (>= min_contributors).
        # Skip only when 2..min_contributors-1: some data but not enough for reliable consensus.
        if n > 1 and n < min_contributors:
            print(f'  Skipping {build_type} aspect={aspect_bucket}: '
                  f'{n} contributor(s) < {min_contributors} required')
            continue

        # Collect all slot data across all contributors
        slot_vectors: dict[str, list[dict]] = defaultdict(list)
        resolutions: list[str] = []
        for iid, entries in contributors.items():
            for e in entries:
                slots = e.get('slots', {})
                for slot_name, geo in slots.items():
                    if isinstance(geo, dict):
                        slot_vectors[slot_name].append(geo)
                if e.get('resolution'):
                    resolutions.append(e['resolution'])

        # Median per slot per component
        aggregated_slots = {}
        for slot_name, geos in slot_vectors.items():
            if len(geos) < 1:
                continue
            def _med(key):
                vals = [g[key] for g in geos if key in g]
                return round(statistics.median(vals), 5) if vals else None
            entry_out = {
                'x0_rel':   _med('x0_rel'),
                'y_rel':    _med('y_rel'),
                'w_rel':    _med('w_rel'),
                'h_rel':    _med('h_rel'),
                'step_rel': _med('step_rel'),
                'count':    round(statistics.median([g['count'] for g in geos if 'count' in g])) if geos else 1,
            }
            if None not in entry_out.values():
                aggregated_slots[slot_name] = entry_out

        if not aggregated_slots:
            continue

        # Pick most common resolution as representative
        rep_res = max(set(resolutions), key=resolutions.count) if resolutions else ''

        results.append({
            'type':          build_type,
            'aspect':        aspect_bucket,
            'res':           rep_res,
            'slots':         aggregated_slots,
            'n_contributors': len(contributors),
            'timestamp':     int(__import__('time').time()),
        })
        print(f'  Community anchor: {build_type} aspect={aspect_bucket} '
              f'({len(contributors)} contributors, {len(aggregated_slots)} slots)')

    return results


def upload_community_anchors(entries: list[dict], models_dir: Path) -> bool:
    """Write community_anchors.json and upload to HF knowledge repo."""
    from huggingface_hub import HfApi, CommitOperationAdd
    from datetime import datetime, timezone
    import io

    payload = {
        'generated_at':  datetime.now(timezone.utc).isoformat() + 'Z',
        'n_contributors': max((e['n_contributors'] for e in entries), default=0),
        'entries':        entries,
    }
    payload_bytes = json.dumps(payload, indent=2, ensure_ascii=False).encode('utf-8')

    # Save locally for reference
    local_path = models_dir / 'community_anchors.json'
    local_path.write_bytes(payload_bytes)

    api = HfApi(token=HF_TOKEN)
    try:
        api.create_commit(
            repo_id=HF_REPO_ID,
            repo_type='dataset',
            operations=[CommitOperationAdd(
                path_in_repo='models/community_anchors.json',
                path_or_fileobj=io.BytesIO(payload_bytes),
            )],
            commit_message=f'community anchors: {len(entries)} entries',
        )
        log.info(f'community_anchors.json uploaded ({len(entries)} entries)')
        return True
    except Exception as e:
        log.error(f'community_anchors upload failed: {e}')
        return False


# ── Screen classifier helpers ─────────────────────────────────────────────────

def _list_screen_type_files(folders: list[str]) -> list[tuple[str, str, str]]:
    """
    Return list of (install_id, stype, sha) for all screen type PNGs in staging.
    Path format: staging/<install_id>/screen_types/<stype>/<sha>.png
    """
    from huggingface_hub import HfApi
    api   = HfApi(token=HF_TOKEN)
    result = []

    # Optimization: instead of listing the whole repo, list each staging/<id>/screen_types
    for iid in folders:
        try:
            path = f'staging/{iid}/screen_types'
            # recursive=True here is fine because we are limited to one user's screen_types
            elements = api.list_repo_tree(HF_DATASET, path_in_repo=path, repo_type='dataset', recursive=True)
            for e in elements:
                # e.path: staging/<iid>/screen_types/<stype>/<sha>.png
                parts = e.path.split('/')
                if len(parts) == 5 and e.path.endswith('.png'):
                    stype = parts[3]
                    sha = parts[4][:-4] # strip .png
                    result.append((iid, stype, sha))
        except Exception:
            # Folder might not exist for this user
            continue

    if result:
        return result

    # Fallback (slow, might timeout on large repos)
    files = list(api.list_repo_files(HF_DATASET, repo_type='dataset'))
    for f in files:
        # staging/<install_id>/screen_types/<stype>/<sha>.png
        parts = f.split('/')
        if len(parts) == 5 and parts[0] == 'staging' and parts[2] == 'screen_types' and f.endswith('.png'):
            install_id = parts[1]
            stype      = parts[3]
            sha        = parts[4][:-4]  # strip .png
            result.append((install_id, stype, sha))
    return result


def _download_screen_shot(install_id: str, stype: str, sha: str, dest_dir: Path) -> Path | None:
    """Download staging/<install_id>/screen_types/<stype>/<sha>.png to dest_dir."""
    from huggingface_hub import hf_hub_download
    dest = dest_dir / f'{sha}.png'
    if dest.exists():
        return dest
    try:
        local = hf_hub_download(
            HF_DATASET,
            f'staging/{install_id}/screen_types/{stype}/{sha}.png',
            repo_type='dataset',
            token=HF_TOKEN,
        )
        import shutil
        shutil.copy2(local, dest)
        return dest
    except Exception as e:
        log.debug(f'Screen shot {sha} from {install_id}/{stype} missing: {e}')
        return None


def collect_screen_type_votes(all_files: list[tuple[str, str, str]]) -> tuple[dict[str, tuple[str, str]], int]:
    """
    Apply democratic voting to screen type screenshots.

    Returns:
        winner_map:  {sha -> (winning_stype, install_id_that_uploaded_it)}
        n_users:     number of install_ids that contributed at least one screenshot
    """
    # sha -> {install_id -> stype}  (each install_id casts 1 vote per sha)
    sha_votes: dict[str, dict[str, str]] = defaultdict(dict)
    sha_source: dict[str, str] = {}

    for install_id, stype, sha in all_files:
        if stype not in SCREEN_TYPES:
            continue
        sha_votes[sha][install_id] = stype
        sha_source.setdefault(sha, install_id)

    n_users = len({iid for iid, _, _ in all_files})

    winner_map: dict[str, tuple[str, str]] = {}
    for sha, votes in sha_votes.items():
        label_counts = Counter(votes.values())
        winner, _ = label_counts.most_common(1)[0]
        winner_map[sha] = (winner, sha_source[sha])

    # Per-class cap: if a screen type has >= SC_MIN_KEEP samples, keep only
    # SC_MAX_KEEP (random selection — avoids bloat for stable UI screens).
    import random as _random
    by_class: dict[str, list[str]] = defaultdict(list)
    for sha, (stype, _) in winner_map.items():
        by_class[stype].append(sha)

    capped: dict[str, tuple[str, str]] = {}
    for stype, shas in by_class.items():
        if len(shas) >= SC_MIN_KEEP and len(shas) > SC_MAX_KEEP:
            _random.shuffle(shas)
            shas = shas[:SC_MAX_KEEP]
            print(f'  Screen type {stype}: capped to {SC_MAX_KEEP} of {len(by_class[stype])} samples')
        for sha in shas:
            capped[sha] = winner_map[sha]

    return capped, n_users


def train_screen_classifier(
    winner_map: dict[str, tuple[str, str]],
    models_dir: Path,
    tmpdir: Path,
    prev_model_pt: Path | None = None,
    deadline: float | None = None,
) -> tuple[float, int]:
    """
    Download winning screenshots, fine-tune MobileNetV3-Small, save to models_dir.
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

    from collections import defaultdict as _dd_sc2

    # ── Collect screenshots (bulk download per install_id) ────────────────────
    # Group by install_id so we can snapshot-download entire screen_types folders
    by_iid_sc: dict[str, list[tuple[str, str]]] = _dd_sc2(list)
    for sha, (stype, iid) in winner_map.items():
        by_iid_sc[iid].append((sha, stype))

    snap_sc = tmpdir / 'snap_sc'
    snap_sc.mkdir(exist_ok=True)

    print(f'\nDownloading {len(winner_map)} screenshots from {len(by_iid_sc)} contributor(s)...')
    import socket as _socket_sc
    import urllib.request as _urllib_sc
    from concurrent.futures import ThreadPoolExecutor as _TPE_sc

    _socket_sc.setdefaulttimeout(120)
    _hf_base_sc = f'https://huggingface.co/datasets/{HF_DATASET}/resolve/main'
    _opener_sc = _urllib_sc.build_opener()
    if HF_TOKEN:
        _opener_sc.addheaders = [('Authorization', f'Bearer {HF_TOKEN}')]

    def _fetch_screen(args: tuple[str, str, str]) -> bool:
        iid, sha, stype = args
        dest = snap_sc / 'staging' / iid / 'screen_types' / stype / f'{sha}.png'
        if dest.exists():
            return True
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            url = f'{_hf_base_sc}/staging/{iid}/screen_types/{stype}/{sha}.png'
            with _opener_sc.open(url) as r:
                dest.write_bytes(r.read())
            return True
        except Exception:
            return False

    _sc_tasks = [(iid, sha, stype) for sha, (stype, iid) in winner_map.items()]
    _sc_ok = _sc_fail = 0
    with _TPE_sc(max_workers=16) as _pool_sc:
        for _r in _pool_sc.map(_fetch_screen, _sc_tasks):
            if _r:
                _sc_ok += 1
            else:
                _sc_fail += 1
    print(f'  {_sc_ok} downloaded, {_sc_fail} failed/skipped.')

    images, labels = [], []
    for iid, items in by_iid_sc.items():
        for sha, stype in items:
            p = snap_sc / 'staging' / iid / 'screen_types' / stype / f'{sha}.png'
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
    unique_labels = sorted(set(labels))
    label_to_idx  = {l: i for i, l in enumerate(unique_labels)}
    idx_to_label  = {i: l for l, i in label_to_idx.items()}
    n_classes     = len(unique_labels)
    y             = [label_to_idx[l] for l in labels]

    print(f'{n_classes} screen type classes: {unique_labels}')

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

    # Stratified split — classes with 1 sample stay in train only
    from collections import defaultdict as _dd_sc
    by_cls_sc: dict[int, list[int]] = _dd_sc(list)
    for i, lbl in enumerate(y):
        by_cls_sc[lbl].append(i)
    train_idx_sc: list[int] = []
    val_idx_sc:   list[int] = []
    for lbl, idxs in by_cls_sc.items():
        random.shuffle(idxs)
        if len(idxs) >= 2:
            val_idx_sc.append(idxs[0])
            train_idx_sc.extend(idxs[1:])
        else:
            train_idx_sc.extend(idxs)
    random.shuffle(train_idx_sc)
    val_idx_sc = val_idx_sc or train_idx_sc[:1]

    ds_train = ScreenDataset([images[i] for i in train_idx_sc], [y[i] for i in train_idx_sc], transform_train)
    ds_val   = ScreenDataset([images[i] for i in val_idx_sc],   [y[i] for i in val_idx_sc],   transform_val)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=SC_BATCH_SIZE, shuffle=True,  num_workers=0)
    dl_val   = torch.utils.data.DataLoader(ds_val,   batch_size=SC_BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training screen_classifier on {device}...')

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

    class _FocalLoss(torch.nn.Module):
        def forward(self, logits, targets):
            ce = _F.cross_entropy(logits, targets, weight=_cw, reduction='none')
            pt = torch.exp(-ce)
            return ((1.0 - pt) ** FOCAL_GAMMA * ce).mean()

    criterion = _FocalLoss().to(device)

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

    print(f'\nscreen_classifier saved — {n_classes} classes, val_acc={best_val_acc:.1%}')
    return best_val_acc, n


# ── Democratic voting ─────────────────────────────────────────────────────────

def collect_votes(staging_folders: list[str]) -> tuple[dict[str, str], dict[str, str], int]:
    """
    Download all staging annotations and apply democratic label voting.

    Returns:
        winner_labels:  {crop_sha256 -> winning_label}
        winner_sources: {crop_sha256 -> install_id_that_uploaded_this_crop}
        n_users:        number of install_ids that contributed at least one crop
    """
    # sha -> {install_id -> label}  (last label wins per install_id)
    sha_votes: dict[str, dict[str, str]] = defaultdict(dict)
    # sha -> install_id (who uploaded this crop file)
    sha_source: dict[str, str] = {}

    n_with_data = 0
    for iid in staging_folders:
        anns = _load_staging_annotations(iid)
        if not anns:
            continue
        n_with_data += 1
        for entry in anns:
            sha   = entry.get('crop_sha256', '').strip()
            label = entry.get('name', '').strip()
            if sha and label:
                sha_votes[sha][iid] = label  # 1 install_id = 1 vote
                sha_source.setdefault(sha, iid)  # record first uploader

    winner_labels: dict[str, str] = {}
    for sha, votes in sha_votes.items():
        # Count votes per label, majority wins; ties go to first encountered
        label_counts = Counter(votes.values())
        winner, _ = label_counts.most_common(1)[0]
        winner_labels[sha] = winner

    return winner_labels, sha_source, n_with_data


# ── Training ──────────────────────────────────────────────────────────────────

def train(winner_labels: dict[str, str], sha_source: dict[str, str],
          models_dir: Path, tmpdir: Path,
          prev_model_pt: Path | None = None,
          deadline: float | None = None) -> tuple[float, int]:
    """
    Download winning crops, train EfficientNet-B0, save model to models_dir.

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
    from collections import defaultdict as _dd3
    from concurrent.futures import ThreadPoolExecutor as _TPE

    # Group by install_id
    by_iid: dict[str, list[tuple[str, str]]] = _dd3(list)
    for sha, label in winner_labels.items():
        iid = sha_source.get(sha)
        if iid:
            by_iid[iid].append((sha, label))

    snap_cache = tmpdir / 'snap'
    snap_cache.mkdir(exist_ok=True)

    _socket.setdefaulttimeout(120)  # 2 min hard timeout per socket read
    _hf_base = f'https://huggingface.co/datasets/{HF_DATASET}/resolve/main'
    _auth_headers = [('Authorization', f'Bearer {HF_TOKEN}')] if HF_TOKEN else []
    _opener = _urllib.build_opener()
    _opener.addheaders = _auth_headers

    def _fetch_crop(args: tuple[str, str]) -> bool:
        iid, sha = args
        dest = snap_cache / 'staging' / iid / 'crops' / f'{sha}.png'
        if dest.exists():
            return True
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with _opener.open(f'{_hf_base}/staging/{iid}/crops/{sha}.png') as r:
                dest.write_bytes(r.read())
            return True
        except Exception:
            return False

    all_crops = [(iid, sha) for iid, items in by_iid.items() for sha, _ in items]
    print(f'\nDownloading {len(all_crops)} crops from {len(by_iid)} contributor(s)...')
    _ok = _fail = 0
    with _TPE(max_workers=16) as _pool:
        for _result in _pool.map(_fetch_crop, all_crops):
            if _result:
                _ok += 1
            else:
                _fail += 1
    print(f'  {_ok} downloaded, {_fail} failed/skipped.')

    crops, labels = [], []
    for iid, items in by_iid.items():
        crop_dir = snap_cache / 'staging' / iid / 'crops'
        for sha, label in items:
            p = crop_dir / f'{sha}.png'
            if not p.exists():
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            crops.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
            labels.append(label)

    print(f'{len(crops)}/{sum(len(v) for v in by_iid.values())} crops loaded.')
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
    print(f'Training on {device}...')

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

    class _FocalLoss(torch.nn.Module):
        def forward(self, logits, targets):
            ce = _F.cross_entropy(logits, targets, weight=_cw, reduction='none')
            pt = torch.exp(-ce)
            return ((1.0 - pt) ** FOCAL_GAMMA * ce).mean()

    criterion = _FocalLoss().to(device)

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

    print(f'\nModel saved — {n_classes} classes, val_acc={best_val_acc:.1%}')
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

    # 1. Find staging folders
    print('\nScanning staging folders...')
    folders = _list_staging_folders()
    if not folders:
        print('No staging contributions found — nothing to do.')
        return
    print(f'Found {len(folders)} contributor(s): {folders[:5]}{"..." if len(folders) > 5 else ""}')

    # 2. Collect votes
    print('\nLoading annotations and computing democratic votes...')
    winner_labels, sha_source, n_users = collect_votes(folders)

    if not winner_labels:
        print('No valid annotations found — nothing to do.')
        return

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

    # Apply min-votes filter
    if args.min > 1:
        print(f'\nApplying min-votes={args.min} filter...')
        from collections import defaultdict as _dd
        sha_vote_counts: dict[str, int] = {}
        sha_votes_full: dict[str, dict[str, str]] = _dd(dict)
        for iid in folders:
            for entry in _load_staging_annotations(iid):
                sha   = entry.get('crop_sha256', '').strip()
                label = entry.get('name', '').strip()
                if sha and label:
                    sha_votes_full[sha][iid] = label
        winner_labels = {
            sha: label
            for sha, label in winner_labels.items()
            if len(sha_votes_full[sha]) >= args.min
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
        val_acc, n_samples = train(winner_labels, sha_source, models_dir, tmpdir,
                                   prev_model_pt=prev_icon_pt, deadline=_train_deadline)

        # Train screen_classifier if data available
        sc_val_acc: float | None = None
        sc_n_samples = 0
        print('\nScanning screen type staging data...')
        try:
            sc_files = _list_screen_type_files(folders)
            print(f'Found {len(sc_files)} screen type screenshot(s) across all users.')
            if sc_files:
                sc_winner_map, sc_n_users = collect_screen_type_votes(sc_files)
                sc_counts = Counter(stype for stype, _ in sc_winner_map.values())
                print(f'{len(sc_winner_map)} unique screenshots, {len(sc_counts)} classes: '
                      + ', '.join(f'{k}={v}' for k, v in sorted(sc_counts.items())))
                if len(sc_winner_map) >= SC_MIN_SAMPLES:
                    print(f'\nTraining MobileNetV3-Small (screen classifier, {sc_n_users} user(s))...')
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

        print('\nUploading models to HF...')
        ok = _upload_model(models_dir, len(label_counts), val_acc, n_samples, n_users,
                           sc_val_acc=sc_val_acc, sc_n_samples=sc_n_samples)
        if ok:
            print(f'\nDone — models published to {HF_REPO_ID}/models/')
        else:
            print('\nERROR — upload failed.', file=sys.stderr)
            sys.exit(1)

        # 5. Community anchors (P11) — aggregate and upload independently of training
        print('\nAggregating community anchors (P11)...')
        anchor_entries = build_community_anchors(folders, min_contributors=3)
        if anchor_entries:
            upload_community_anchors(anchor_entries, models_dir)
        else:
            print('Not enough contributors for community anchors yet (need >= 3).')


if __name__ == '__main__':
    main()
