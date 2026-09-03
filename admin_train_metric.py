#!/usr/bin/env python3
"""
admin_train_metric.py — Metric-learning variant of icon classifier
====================================================================
Same data pipeline as admin_train.py (democratic voting, HF staging download),
but replaces the softmax classification head with an ArcFace metric-learning
head. Output is an embedder + k-NN gallery instead of a softmax classifier.

Why: classes have a long-tail distribution (48% of classes have a single crop;
top class has 339). Softmax learns frequent classes well and barely learns
rare ones. ArcFace + PK sampler trains the model to make same-class crops
close in embedding space and different-class crops far apart — every class
gets equal training signal regardless of count, and 1-sample classes still
work via nearest-neighbour lookup.

Outputs (next to icon_classifier.pt, not replacing it):
    icon_embedder.pt           — backbone + projection head state_dict
    icon_embedder_meta.json    — embed_dim, input_size, val_recall@1, ...
    embedding_index.npz        — gallery embeddings of all training crops
    label_map.json             — int → label name (same format as classifier)

Run from sets-warp dir:
    .venv/bin/python ../sets-warp-backend/admin_train_metric.py --train --min 1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Reuse helpers from admin_train.py — single source of truth for the data pipeline
sys.path.insert(0, str(Path(__file__).parent))
from admin_train import (  # noqa: E402
    HF_TOKEN, HF_DATASET, HF_REPO_ID,
    IMG_SIZE, MODEL_IMG_SIZE,
    _require_hf, read_curated_crops,
    _create_commit_with_retry,
)
import hashlib  # noqa: E402

UTC = timezone.utc

# ── Metric-learning hyper-parameters ─────────────────────────────────────────

EMBED_DIM    = 256
ARC_MARGIN   = 0.5    # ArcFace angular margin (radians) — 0.5 is the paper default
ARC_SCALE    = 30.0   # ArcFace logit scale — temperature for softmax

# PK sampler: each batch is P classes × K samples
PK_P = 8
PK_K = 4
BATCH_SIZE = PK_P * PK_K  # 32

MAX_EPOCHS    = 30
LR            = 3e-4
PATIENCE      = 5
BATCHES_PER_EPOCH_MIN = 60  # ensures small datasets still get enough updates
MIN_SAMPLES   = 5
N_FEATURE_AUGS = 5   # augmented backbone-feature passes for training crops


# ── ArcFace head ─────────────────────────────────────────────────────────────

def _build_arcface_head(embed_dim: int, n_classes: int):
    """ArcFace head: cosine similarity between L2-normed embeddings and L2-normed
    class weights, with an angular margin added to the target class during training."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ArcFaceHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.W = nn.Parameter(torch.empty(n_classes, embed_dim))
            nn.init.xavier_normal_(self.W)
            self.margin = ARC_MARGIN
            self.scale = ARC_SCALE

        def forward(self, emb, labels=None):
            # emb: (B, D) already L2-normalized
            W = F.normalize(self.W, dim=1)
            cos = emb @ W.t()
            if labels is None:
                return cos * self.scale
            # add margin only to target class: cos(theta + m)
            cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(cos)
            target = torch.cos(theta + self.margin)
            one_hot = F.one_hot(labels, num_classes=cos.shape[1]).float()
            logits = cos * (1 - one_hot) + target * one_hot
            return logits * self.scale

    return ArcFaceHead()


# ── PK sampler ───────────────────────────────────────────────────────────────

class PKBatchSampler:
    """Yields batches of P classes × K samples each. Classes with <K samples are
    sampled with replacement (rare classes still get equal class-level airtime)."""
    def __init__(self, labels: list[int], P: int, K: int, num_batches: int):
        self.P = P
        self.K = K
        self.num_batches = num_batches
        self.label_to_idx: dict[int, list[int]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            self.label_to_idx[lbl].append(i)
        self.classes = list(self.label_to_idx.keys())

    def __iter__(self):
        for _ in range(self.num_batches):
            picked = random.sample(self.classes, min(self.P, len(self.classes)))
            batch = []
            for c in picked:
                pool = self.label_to_idx[c]
                if len(pool) >= self.K:
                    batch.extend(random.sample(pool, self.K))
                else:
                    batch.extend(random.choices(pool, k=self.K))
            yield batch

    def __len__(self):
        return self.num_batches


# ── Pre-computed feature helpers ─────────────────────────────────────────────

class _FeatureDataset:
    """Wraps pre-computed (feature, label) tensors for use with DataLoader."""
    def __init__(self, features, labels):
        self.features, self.labels = features, labels
    def __len__(self): return len(self.features)
    def __getitem__(self, i): return self.features[i], self.labels[i]


def _extract_features(backbone, crops_list, labels_list, transform, device,
                      n_passes=1, batch_size=64):
    """Run crops through frozen backbone.  Multiple passes with a random-aug
    transform produce diverse feature vectors for training."""
    import cv2
    import torch

    class _ImgDS(torch.utils.data.Dataset):
        def __init__(self, c, l, tf):
            self.c, self.l, self.tf = c, l, tf
        def __len__(self): return len(self.c)
        def __getitem__(self, i):
            return self.tf(cv2.cvtColor(self.c[i], cv2.COLOR_BGR2RGB)), self.l[i]

    backbone.eval()
    feats, lbls = [], []
    for p in range(n_passes):
        dl = torch.utils.data.DataLoader(
            _ImgDS(crops_list, labels_list, transform),
            batch_size=batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            for xb, yb in dl:
                feats.append(backbone(xb.to(device)).cpu())
                lbls.append(yb)
        if n_passes > 1:
            print(f'    pass {p+1}/{n_passes}')
    return torch.cat(feats), torch.cat(lbls)


def _embed_features(proj, features, device, batch_size=512):
    """Run pre-computed backbone features through the projection head.
    Returns L2-normalised embeddings as a numpy float32 array."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    proj.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = features[i:i + batch_size].to(device)
            embs.append(F.normalize(proj(batch), dim=1).cpu().numpy())
    return np.concatenate(embs, axis=0).astype('float32')


# ── Embedding gallery ────────────────────────────────────────────────────────

def _build_gallery(model, head, ds_train, device, batch_size: int = 64):
    """Embed all training crops with model.eval() (no aug). Returns
    (embeddings: np.ndarray (N, D) float32, labels: np.ndarray (N,) int32)."""
    import numpy as np
    import torch

    model.eval()
    head.eval()
    embs, lbls = [], []
    dl = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            e = model(xb)            # (B, D) already L2-normed by projection
            embs.append(e.cpu().numpy())
            lbls.append(yb.numpy())
    return (
        np.concatenate(embs, axis=0).astype('float32'),
        np.concatenate(lbls, axis=0).astype('int32'),
    )


def _recall_at_1(gallery_emb, gallery_lbl, query_emb, query_lbl) -> float:
    """Top-1 nearest-neighbour accuracy of queries against gallery."""
    import numpy as np
    if len(query_emb) == 0:
        return 0.0
    sims = query_emb @ gallery_emb.T  # cosine (both L2-normed)
    top = np.argmax(sims, axis=1)
    pred = gallery_lbl[top]
    return float((pred == query_lbl).mean())


# ── Backbone + projection ────────────────────────────────────────────────────

def _build_embedder(prev_model_pt: Path | None = None):
    """EfficientNet-B0 backbone + Linear → L2-normalize projection to EMBED_DIM."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as tv_models

    backbone = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()  # drop the softmax head; raw pooled features

    class Embedder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.proj = nn.Linear(in_features, EMBED_DIM)

        def forward(self, x):
            f = self.backbone(x)
            e = self.proj(f)
            return F.normalize(e, dim=1)

    model = Embedder()

    # Warm-start: load backbone weights from the previous central softmax model
    if prev_model_pt and prev_model_pt.exists():
        try:
            state = torch.load(str(prev_model_pt), map_location='cpu')
            # admin_train.py saves the full softmax model — keep only feature layers
            backbone_state = {}
            for k, v in state.items():
                if k.startswith('features.') or k.startswith('avgpool.'):
                    backbone_state[f'backbone.{k}'] = v
            missing, unexpected = model.load_state_dict(backbone_state, strict=False)
            n_loaded = sum(1 for k in state if k.startswith('features.') or k.startswith('avgpool.'))
            print(f'Warm-start: loaded {n_loaded} backbone tensors from {prev_model_pt.name}')
        except Exception as e:
            print(f'Warm-start failed ({e}) — using ImageNet weights')
    else:
        print('No previous model — starting from ImageNet weights')

    return model


# ── Main training function ───────────────────────────────────────────────────

def train_metric(winner_labels: dict[str, str],
                 models_dir: Path, tmpdir: Path,
                 prev_model_pt: Path | None = None,
                 deadline: float | None = None) -> tuple[float, int]:
    """Mirror of admin_train.train() but with ArcFace + PK sampler.
    Reads crops from data/crops/<sha>.png (curated by
    democratic_merge_crops.py). Returns (val_recall@1, n_samples_used)."""
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T

    # ── Crop download (mirrors admin_train.train()) ──────────────────────────
    import socket as _socket
    import urllib.request as _urllib
    from concurrent.futures import ThreadPoolExecutor as _TPE

    snap_cache = tmpdir / 'snap'
    snap_cache.mkdir(exist_ok=True)
    _socket.setdefaulttimeout(120)
    _hf_base = f'https://huggingface.co/datasets/{HF_DATASET}/resolve/main'
    _opener = _urllib.build_opener()
    _opener.addheaders = [('Authorization', f'Bearer {HF_TOKEN}')] if HF_TOKEN else []

    # One git clone, not 12 274 HTTP requests.
    #
    # The REST path is rate-limited: fetching the crops one by one drew
    # `HTTP 429 Too Many Requests` and the run finished with 4 520 of 12 274
    # crops, so the embedder trained on a third of the dataset and published
    # — 1 637 classes against 2 867. The client hit the same wall on cold
    # start (`_MAX_DOWNLOAD_WORKERS = 3`, "HF rate-limits anonymous parallel
    # HEADs"), and `tools/build_crops_tarball.py` in sto-warp already fetches
    # exactly this data over the git protocol for exactly this reason.
    #
    # `git lfs` pulls the pixels; a shallow sparse checkout keeps it to
    # `data/crops`. One operation, no per-file throttling, and it either
    # succeeds or fails loudly.
    import subprocess as _sp
    from hf_clone import _redact
    _crops_root = snap_cache / 'data' / 'crops'
    _clone_dir  = tmpdir / 'crops_repo'
    _clone_url  = f'https://huggingface.co/datasets/{HF_DATASET}'
    # The token goes in a header, never in the URL or in `.git/config`, and
    # any failure is re-raised without it: `CalledProcessError` prints the
    # whole command line, and these tools run in a maintainer's terminal as
    # well as on CI.
    _auth = [] if not HF_TOKEN else [
        '-c', f'http.extraHeader=Authorization: Bearer {HF_TOKEN}']

    def _git(*args, cwd=None):
        try:
            _sp.run(['git', *args], check=True,
                    cwd=str(cwd) if cwd else None)
        except _sp.CalledProcessError as exc:
            raise _sp.CalledProcessError(
                exc.returncode, [_redact(a, HF_TOKEN) for a in exc.cmd],
                output=exc.output, stderr=exc.stderr) from None

    print('Cloning data/crops over git (the REST path is rate-limited)...')
    _sp.run(['git', 'lfs', 'install'], check=True,
            stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
    _git(*_auth, 'clone', '--no-checkout', '--depth', '1',
         _clone_url, str(_clone_dir))
    _git('sparse-checkout', 'set', 'data/crops', cwd=_clone_dir)
    _git(*_auth, 'checkout', cwd=_clone_dir)

    # Flatten both layouts into the cache the trainer reads: `data/crops/` is
    # part sharded (`<ab>/<sha>.png`, HF caps a directory at 10 000 files)
    # and part flat, from before the shards.
    _crops_root.mkdir(parents=True, exist_ok=True)
    _cloned = 0
    for _png in (_clone_dir / 'data' / 'crops').rglob('*.png'):
        _dst = _crops_root / _png.name
        if not _dst.exists():
            _dst.write_bytes(_png.read_bytes())
        _cloned += 1
    print(f'  {_cloned} crop(s) checked out.')
    _first_errors: list[str] = []

    def _fetch_crop(sha: str) -> bool:
        dest = snap_cache / 'data' / 'crops' / f'{sha}.png'
        if dest.exists():
            return True
        # Not in the checkout: the annotations reference a crop the repo does
        # not carry. Reported rather than counted — a third of the dataset
        # once went missing behind a bare number.
        if len(_first_errors) < 5:
            _first_errors.append(f'{sha[:12]}: not in data/crops/')
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
    if _fail:
        for _e in _first_errors:
            print(f'    first failures — {_e}')

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

    return _fit_metric(crops, labels, models_dir, prev_model_pt, deadline)


def _fit_metric(crops: list, labels: list[str],
                models_dir: Path,
                prev_model_pt: Path | None,
                deadline: float | None) -> tuple[float, int]:
    """Core training loop — pre-computes backbone features for CPU efficiency.
    On CPU, running EfficientNet-B0 forward+backward each epoch is too slow for
    CI timeouts.  Instead: extract backbone features once (with N augmented
    passes), then train only the lightweight projection + ArcFace head on
    cached features.  Backbone weights are preserved in the saved model so
    inference still runs the full pipeline (image → backbone → proj → L2)."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T

    n = len(crops)
    print(f'{n} crops ready.')
    if n < MIN_SAMPLES:
        raise RuntimeError(f'Only {n} crops available (need {MIN_SAMPLES}).')

    # ── Label map ────────────────────────────────────────────────────────────
    unique_labels = sorted(set(labels))
    label_to_idx  = {l: i for i, l in enumerate(unique_labels)}
    idx_to_label  = {i: l for l, i in label_to_idx.items()}
    n_classes     = len(unique_labels)
    y             = [label_to_idx[l] for l in labels]
    print(f'{n_classes} classes')

    # ── Transforms ───────────────────────────────────────────────────────────
    transform_train = T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop(MODEL_IMG_SIZE, scale=(0.8, 1.0)),
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

    # ── Stratified train/val split ───────────────────────────────────────────
    by_cls: dict[int, list[int]] = defaultdict(list)
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

    train_crops  = [crops[i] for i in train_idx]
    train_labels = [y[i] for i in train_idx]
    val_crops    = [crops[i] for i in val_idx]
    val_labels   = [y[i] for i in val_idx]

    # ── Model ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _build_embedder(prev_model_pt).to(device)

    # ── Pre-compute backbone features ────────────────────────────────────────
    # The EfficientNet-B0 forward pass dominates wall-time on CPU.  Extracting
    # features once (with N_FEATURE_AUGS augmented passes for diversity) and
    # then training only proj + ArcFace head on cached 1280-d vectors cuts
    # total training from >2 h to ~20-30 min.
    n_aug = N_FEATURE_AUGS

    print(f'\n── icon_embedder (EfficientNet-B0 + ArcFace) {"─" * 28}')
    print(f'  Dataset : {n} crops, {n_classes} classes ({len(train_idx)} train, {len(val_idx)} val)')
    print(f'  Embed   : {EMBED_DIM}-d, margin={ARC_MARGIN}, scale={ARC_SCALE}')
    print(f'  Device  : {device}')
    print(f'  Strategy: pre-compute backbone features ({n_aug} aug passes)')
    print(f'{"─" * 64}')

    print(f'  Extracting backbone features...')
    t0 = time.monotonic()

    train_aug_feats, train_aug_labels = _extract_features(
        model.backbone, train_crops, train_labels,
        transform_train, device, n_passes=n_aug)

    train_eval_feats, train_eval_labels = _extract_features(
        model.backbone, train_crops, train_labels,
        transform_val, device, n_passes=1)

    val_feats, val_feat_labels = _extract_features(
        model.backbone, val_crops, val_labels,
        transform_val, device, n_passes=1)

    all_feats, all_feat_labels = _extract_features(
        model.backbone, crops, y,
        transform_val, device, n_passes=1)

    dt = time.monotonic() - t0
    print(f'  Features extracted in {dt:.0f}s: '
          f'{len(train_aug_feats)} train (x{n_aug} aug), '
          f'{len(val_feats)} val')

    # Free backbone from compute device — only proj + head needed now
    model.backbone.cpu()

    # ── PK sampler on cached features ────────────────────────────────────────
    train_feat_ds = _FeatureDataset(train_aug_feats, train_aug_labels)
    batches_per_epoch = max(BATCHES_PER_EPOCH_MIN, len(train_feat_ds) // BATCH_SIZE)
    pk_sampler = PKBatchSampler(
        train_aug_labels.tolist(), P=PK_P, K=PK_K,
        num_batches=batches_per_epoch)
    dl_train = torch.utils.data.DataLoader(
        train_feat_ds, batch_sampler=pk_sampler, num_workers=0)

    print(f'  Sampler : P={PK_P} x K={PK_K} -> batch={BATCH_SIZE}, '
          f'{batches_per_epoch} batches/epoch')

    # ── ArcFace head + optimiser (proj + head only) ──────────────────────────
    head = _build_arcface_head(EMBED_DIM, n_classes).to(device)
    optimizer = torch.optim.AdamW(
        list(model.proj.parameters()) + list(head.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # ── Training loop (proj + head only — seconds per epoch) ─────────────────
    best_recall    = 0.0
    best_state     = None
    patience_count = 0

    for epoch in range(MAX_EPOCHS):
        if deadline is not None and time.monotonic() > deadline:
            print(f'  Time budget exceeded, stopping at epoch {epoch+1}.')
            break

        model.proj.train(); head.train()
        loss_sum = 0.0
        n_batch = 0
        for feat_batch, yb in dl_train:
            feat_batch, yb = feat_batch.to(device), yb.to(device)
            optimizer.zero_grad()
            emb = F.normalize(model.proj(feat_batch), dim=1)
            logits = head(emb, yb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n_batch += 1
        scheduler.step()
        avg_loss = loss_sum / max(1, n_batch)

        # Validation: k-NN recall@1 on pre-computed features
        model.proj.eval()
        g_emb = _embed_features(model.proj, train_eval_feats, device)
        q_emb = _embed_features(model.proj, val_feats, device)
        val_recall = _recall_at_1(
            g_emb, train_eval_labels.numpy(),
            q_emb, val_feat_labels.numpy())
        print(f'  Epoch {epoch+1:2d}/{MAX_EPOCHS}  loss={avg_loss:.3f}  '
              f'val_recall@1={val_recall:.1%}  best={best_recall:.1%}')

        if val_recall > best_recall:
            best_recall = val_recall
            best_state = {
                'proj': {k: v.cpu().clone() for k, v in model.proj.state_dict().items()},
                'head': {k: v.cpu().clone() for k, v in head.state_dict().items()},
            }
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f'  Early stop at epoch {epoch+1}.')
                break

    if best_state:
        model.proj.load_state_dict(best_state['proj'])
        head.load_state_dict(best_state['head'])

    # ── Save ─────────────────────────────────────────────────────────────────
    models_dir.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()

    # Final gallery from pre-computed features + trained projection
    full_emb = _embed_features(model.proj, all_feats, torch.device('cpu'))
    full_lbl = all_feat_labels.numpy().astype('int32')

    torch.save(model.state_dict(), str(models_dir / 'icon_embedder.pt'))
    np.savez(
        str(models_dir / 'embedding_index.npz'),
        embeddings=full_emb,
        labels=full_lbl,
    )
    with open(models_dir / 'embedder_label_map.json', 'w', encoding='utf-8') as f:
        json.dump(idx_to_label, f, ensure_ascii=False, indent=2)
    with open(models_dir / 'icon_embedder_meta.json', 'w', encoding='utf-8') as f:
        json.dump({
            'n_classes':    n_classes,
            'embed_dim':    EMBED_DIM,
            'input_size':   MODEL_IMG_SIZE,
            'val_recall@1': best_recall,
            'gallery_size': len(full_emb),
            'arc_margin':   ARC_MARGIN,
            'arc_scale':    ARC_SCALE,
            'pk_p':         PK_P,
            'pk_k':         PK_K,
            'n_feature_augs': n_aug,
            'trained_at':   datetime.now(UTC).isoformat() + 'Z',
        }, f, indent=2)

    print(f'\n✓ icon_embedder saved — {n_classes} classes, '
          f'val_recall@1={best_recall:.1%}, gallery={len(full_emb)}')
    return best_recall, n


# ── HF upload ────────────────────────────────────────────────────────────────

def _upload_embedder(models_dir: Path) -> bool:
    """Upload icon_embedder.pt + label_map.json + embedding_index.npz + meta to
    sets-sto/warp-knowledge under models/. Mirrors admin_train._upload_model().
    Files are placed alongside (not replacing) the softmax classifier files."""
    from huggingface_hub import HfApi, CommitOperationAdd

    pt   = models_dir / 'icon_embedder.pt'
    lbl  = models_dir / 'embedder_label_map.json'
    idx  = models_dir / 'embedding_index.npz'
    meta = models_dir / 'icon_embedder_meta.json'
    if not pt.exists():
        print('ERROR: icon_embedder.pt not found — nothing to upload')
        return False

    sha = hashlib.sha256(pt.read_bytes()).hexdigest()[:16]
    print(f'Uploading embedder to {HF_REPO_ID} (version={sha})...')

    api = HfApi(token=HF_TOKEN)
    ops = [
        CommitOperationAdd(path_in_repo='models/icon_embedder.pt',         path_or_fileobj=str(pt)),
        CommitOperationAdd(path_in_repo='models/embedding_index.npz',      path_or_fileobj=str(idx)),
        CommitOperationAdd(path_in_repo='models/icon_embedder_meta.json',  path_or_fileobj=str(meta)),
        CommitOperationAdd(path_in_repo='models/embedder_label_map.json',  path_or_fileobj=str(lbl)),
    ]
    commit_msg = f'admin_train_metric: icon_embedder version={sha}'
    ok = _create_commit_with_retry(api, HF_REPO_ID, 'dataset', ops, commit_msg)
    if ok:
        print(f'Uploaded embedder version={sha}')
    return ok


# ── CLI ──────────────────────────────────────────────────────────────────────

_LOCAL_NAME_RE = __import__('re').compile(
    r'^([^_]+(?:_[^_]+)*?)__(.+)__[0-9a-f]+\.png$', __import__('re').IGNORECASE
)


def _main_local(args):
    """Train from local crops/ directory — skips HF entirely.
    Filename convention: <prefix>__<label>__<hash>.png (matches WARP CORE crops)."""
    import cv2
    import tempfile

    crop_dir = Path(args.local_crops)
    if not crop_dir.is_dir():
        print(f'ERROR: --local-crops dir does not exist: {crop_dir}')
        sys.exit(2)

    print('=' * 60)
    print('WARP Metric-Learning Trainer (LOCAL MODE)')
    print(f'Crops:   {crop_dir}')
    print('=' * 60)

    crops, labels = [], []
    skipped = 0
    for p in sorted(crop_dir.glob('*.png')):
        m = _LOCAL_NAME_RE.match(p.name)
        if not m:
            skipped += 1
            continue
        img = cv2.imread(str(p))
        if img is None:
            skipped += 1
            continue
        crops.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        labels.append(m.group(2))
    print(f'\nLoaded {len(crops)} crops, skipped {skipped}')

    out_dir = Path(args.out) if args.out else (
        Path(__file__).parent.parent / 'sets-warp' / 'warp' / 'models'
    )
    prev_pt = Path(args.warm_start_from) if args.warm_start_from else (
        out_dir / 'icon_classifier.pt'
    )
    print(f'Output dir: {out_dir}')
    if prev_pt.exists():
        print(f'Warm-start: {prev_pt.name}')

    _fit_metric(
        crops, labels,
        models_dir=out_dir,
        prev_model_pt=prev_pt if prev_pt.exists() else None,
        deadline=None,
    )


def main():
    parser = argparse.ArgumentParser(
        description='WARP metric-learning trainer (ArcFace + EfficientNet-B0 + k-NN)',
    )
    parser.add_argument('--train', action='store_true', help='Train and save model')
    parser.add_argument('--min', type=int, default=1, metavar='N',
                        help='Minimum unique users per crop label (default: 1)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output directory (default: <sets-warp>/warp/models)')
    parser.add_argument('--warm-start-from', type=str, default=None,
                        help='Path to previous icon_classifier.pt for backbone warm-start')
    parser.add_argument('--local-crops', type=str, default=None,
                        help='Skip HF download — train on a local crops/ directory. '
                             'Filenames must follow <prefix>__<label>__<hash>.png.')
    parser.add_argument('--upload', action='store_true',
                        help='Upload trained embedder to HF after training')
    parser.add_argument('--deadline-minutes', type=int, default=None, metavar='M',
                        help='Stop training after M minutes (leaves time for upload in CI)')
    args = parser.parse_args()

    # ── Local crops mode (fast iteration) ────────────────────────────────────
    if args.local_crops:
        return _main_local(args)

    _require_hf()

    print('=' * 60)
    print('WARP Metric-Learning Trainer (ArcFace)')
    print(f'Dataset: {HF_DATASET}')
    print(f'Mode:    {"TRAIN" if args.train else "DRY-RUN"}')
    print(f'Min votes: {args.min}')
    print('=' * 60)

    print('\nReading curated crop labels from data/annotations.jsonl...')
    winner_labels, vote_counts = read_curated_crops()
    if not winner_labels:
        print('No curated annotations.')
        return
    peak_votes = max(vote_counts.values(), default=0)
    print(f'  {len(winner_labels)} crops, peak votes/crop={peak_votes}')

    if not args.train:
        print('\nDry-run finished. Use --train to actually train.')
        return

    # Default output: writes directly to the sets-warp models directory so the
    # next WARP run picks up the new embedder without manual file copying.
    out_dir = Path(args.out) if args.out else (
        Path(__file__).parent.parent / 'sets-warp' / 'warp' / 'models'
    )
    print(f'\nOutput dir: {out_dir}')

    prev_pt = Path(args.warm_start_from) if args.warm_start_from else (
        out_dir / 'icon_classifier.pt'
    )

    _deadline = (time.monotonic() + args.deadline_minutes * 60
                 if args.deadline_minutes else None)

    with tempfile.TemporaryDirectory(prefix='warp_metric_') as td:
        train_metric(
            winner_labels,
            models_dir=out_dir, tmpdir=Path(td),
            prev_model_pt=prev_pt if prev_pt.exists() else None,
            deadline=_deadline,
        )

    if args.upload:
        _upload_embedder(out_dir)


if __name__ == '__main__':
    main()
