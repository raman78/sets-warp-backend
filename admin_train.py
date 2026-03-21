#!/usr/bin/env python3
"""
admin_train.py — WARP Central Model Trainer
============================================
Downloads all user-contributed staging crops from sets-sto/sto-icon-dataset,
applies democratic label voting (1 install_id = 1 vote, majority per crop),
trains EfficientNet-B0, uploads the model to sets-sto/warp-knowledge.

Run with the sets-warp .venv (has torch, torchvision, cv2, huggingface_hub):
    /path/to/sets-warp/.venv/bin/python admin_train.py
    /path/to/sets-warp/.venv/bin/python admin_train.py --train

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
    api   = HfApi(token=HF_TOKEN)
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
                  n_samples: int, n_users: int) -> bool:
    """Upload model files to sets-sto/warp-knowledge under models/."""
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi(token=HF_TOKEN)

    pt_path     = models_dir / 'icon_classifier.pt'
    label_path  = models_dir / 'label_map.json'
    meta_path   = models_dir / 'icon_classifier_meta.json'

    if not pt_path.exists():
        log.error('Model file not found — nothing to upload')
        return False

    # Compute version hash (sha256 of model file, first 16 hex chars)
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

    try:
        api.create_commit(
            repo_id=HF_REPO_ID,
            repo_type='dataset',
            operations=ops,
            commit_message=(
                f'admin_train: {n_classes} classes, val_acc={val_acc:.1%}, '
                f'{n_samples} samples from {n_users} users ({trained_at[:10]})'
            ),
        )
        log.info(f'Model uploaded to {HF_REPO_ID}: version={sha}, val_acc={val_acc:.1%}')
        return True
    except Exception as e:
        log.error(f'Upload failed: {e}')
        return False


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
          models_dir: Path, tmpdir: Path) -> tuple[float, int]:
    """
    Download winning crops, train EfficientNet-B0, save model to models_dir.

    Returns (best_val_acc, n_samples_used).
    """
    import cv2
    import torch
    import torchvision.models as tv_models
    import torchvision.transforms as T
    import torch.nn.functional as _F
    import random
    from collections import Counter as _Counter

    # ── Collect crops (parallel download) ────────────────────────────────────
    from concurrent.futures import ThreadPoolExecutor, as_completed

    items_to_dl = [
        (sha, label, sha_source[sha])
        for sha, label in winner_labels.items()
        if sha_source.get(sha)
    ]
    print(f'\nDownloading {len(items_to_dl)} crops (16 threads)...')

    def _dl(args):
        sha, label, iid = args
        path = _download_crop(iid, sha, tmpdir)
        if path is None:
            return None
        img = cv2.imread(str(path))
        if img is None:
            return None
        return cv2.resize(img, (IMG_SIZE, IMG_SIZE)), label

    crops, labels = [], []
    done = 0
    total_dl = len(items_to_dl)
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(_dl, item): item for item in items_to_dl}
        for fut in as_completed(futs):
            done += 1
            result = fut.result()
            if result is not None:
                crops.append(result[0])
                labels.append(result[1])
            if done % 100 == 0 or done == total_dl:
                print(f'  {done}/{total_dl} done, {len(crops)} loaded')

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
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
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

    idx_all = list(range(n))
    random.shuffle(idx_all)
    split      = max(1, int(n * 0.8))
    train_idx  = idx_all[:split]
    val_idx    = idx_all[split:] or idx_all[:1]

    ds_train = CropDataset([crops[i] for i in train_idx], [y[i] for i in train_idx], transform_train)
    ds_val   = CropDataset([crops[i] for i in val_idx],   [y[i] for i in val_idx],   transform_val)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    dl_val   = torch.utils.data.DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}...')

    model = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, n_classes)
    model = model.to(device)

    if n < 50:
        for p in model.features.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
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

    # 2b. Skip-if-unchanged check (fast path before downloading crops)
    if args.skip_if_unchanged and args.train:
        current_shas = set(winner_labels.keys())
        last_shas    = _load_training_manifest()
        if last_shas and current_shas == last_shas:
            print(f'\nNo new crops since last training ({len(current_shas)} crops unchanged) — skipping.')
            return
        new_count = len(current_shas - last_shas)
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

        print('\nTraining EfficientNet-B0...')
        val_acc, n_samples = train(winner_labels, sha_source, models_dir, tmpdir)

        # 4. Upload
        # Save training manifest (so next run can skip if nothing changed)
        _save_training_manifest(set(winner_labels.keys()), models_dir)

        print('\nUploading model to HF...')
        ok = _upload_model(models_dir, len(label_counts), val_acc, n_samples, n_users)
        if ok:
            print(f'\nDone — model published to {HF_REPO_ID}/models/')
        else:
            print('\nERROR — upload failed.', file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    main()
