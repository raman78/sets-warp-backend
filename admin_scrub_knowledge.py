#!/usr/bin/env python3
"""
admin_scrub_knowledge.py — Remove poisoned entries from knowledge.json
========================================================================
One-shot cleanup tool. Identifies pHash entries in the community
knowledge.json whose contributing crops are actually blank/inactive
slots (virtual classes) but were uploaded under a real ability name —
the classic pre-bootstrap poisoning, e.g. blank slot pHashes mapped to
"Charged Particle Burst" because the old embedder mis-classified blanks
and a contributor confirmed the wrong label in WARP CORE.

Why this is separate from admin_merge.py's _is_poison_name():
    _is_poison_name only catches labels starting with '__' or
    'Test Item Name'. It does NOT catch a poisoned entry whose label is
    a legitimate ability name. We need ground truth — the bootstrapped
    embedder — to tell us the original crops were actually blanks.

Algorithm:
    1. Download knowledge.json + the bootstrapped embedder from HF.
    2. Shallow-clone the dataset to enumerate contributions/<date>/<id>.json
       and build phash → [contribution_id, ...].
    3. For each (phash, name) entry in knowledge.json that has surviving
       contributions, fetch one sample PNG and classify via embedder.
       Drop the entry iff embedder returns a virtual class
       (__empty__ / __inactive__) with cos-sim >= VIRTUAL_CONF.
    4. Dry-run report by default; --apply writes cleaned knowledge.json
       back to HF.

Run from sets-warp-backend dir:
    .venv/bin/python admin_scrub_knowledge.py             # dry-run
    .venv/bin/python admin_scrub_knowledge.py --apply     # write back
    .venv/bin/python admin_scrub_knowledge.py --limit 50  # sample first 50

Environment variables (.env, same as admin_merge.py):
    HF_TOKEN     — HF write token
    HF_REPO_ID   — default: sets-sto/warp-knowledge
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

UTC = timezone.utc


# ── .env loader (mirror admin_merge.py) ──────────────────────────────────────

def _load_env():
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()

HF_TOKEN   = os.environ.get('HF_TOKEN', '')
HF_REPO_ID = os.environ.get('HF_REPO_ID', 'sets-sto/warp-knowledge')

# Threshold for accepting an embedder verdict that a crop is virtual.
# Mirrors VIRTUAL_OVERRIDE_CONF in warp/recognition/icon_matcher.py — both
# layers consult the same model, so they should use the same threshold.
VIRTUAL_CONF = 0.40
VIRTUAL_NAMES = {'__empty__', '__inactive__'}


# ── HF I/O ────────────────────────────────────────────────────────────────────

def _require_hf() -> None:
    if not HF_TOKEN:
        print('ERROR: HF_TOKEN not set', file=sys.stderr)
        sys.exit(2)


def _download_knowledge() -> dict:
    """Download the current knowledge.json payload (full envelope)."""
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(
        HF_REPO_ID, 'knowledge.json',
        repo_type='dataset', token=HF_TOKEN or None,
    )
    return json.loads(Path(local).read_text(encoding='utf-8'))


def _download_embedder_assets(dest: Path) -> None:
    """Pull icon_embedder.pt + embedding_index.npz + embedder_label_map.json
    into dest/. These live under models/ in the knowledge dataset."""
    from huggingface_hub import hf_hub_download
    for fname in ('icon_embedder.pt',
                  'embedding_index.npz',
                  'embedder_label_map.json'):
        src = hf_hub_download(
            HF_REPO_ID, f'models/{fname}',
            repo_type='dataset', token=HF_TOKEN or None,
        )
        (dest / fname).write_bytes(Path(src).read_bytes())


def _clone_contributions() -> Path:
    """Shallow-clone the knowledge dataset; return the working-tree path.
    JSON contribution metadata is fetched in one round-trip; PNGs may be
    LFS pointers (we hf_hub_download them on demand)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from hf_clone import clone_hf_shallow  # noqa: E402
    return clone_hf_shallow(HF_REPO_ID, HF_TOKEN, repo_type='dataset')


def _build_phash_index(snap_dir: Path) -> dict[str, list[Path]]:
    """Walk contributions/<date>/<id>.json and group by pHash.
    Returns {phash: [contribution_json_path, ...]}."""
    contrib_root = snap_dir / 'contributions'
    if not contrib_root.exists():
        log.warning(f'no contributions/ folder at {contrib_root}')
        return {}
    index: dict[str, list[Path]] = defaultdict(list)
    for jp in contrib_root.glob('*/*.json'):
        try:
            rec = json.loads(jp.read_text(encoding='utf-8'))
        except Exception:
            continue
        ph = rec.get('phash', '').strip()
        if ph:
            index[ph].append(jp)
    return index


def _fetch_contribution_png(json_path: Path) -> bytes | None:
    """Resolve the .png next to the .json contribution. The clone may
    have it as a real blob OR an LFS pointer — in the latter case,
    fetch the actual blob via hf_hub_download."""
    png_local = json_path.with_suffix('.png')
    if png_local.exists():
        data = png_local.read_bytes()
        if data.startswith(b'\x89PNG'):
            return data
        # LFS pointer file — fall through to remote fetch.

    rel = png_local.relative_to(png_local.parents[2])  # contributions/<date>/<id>.png
    try:
        from huggingface_hub import hf_hub_download
        remote = hf_hub_download(
            HF_REPO_ID, str(rel),
            repo_type='dataset', token=HF_TOKEN or None,
        )
        data = Path(remote).read_bytes()
        if data.startswith(b'\x89PNG'):
            return data
    except Exception as e:
        log.debug(f'png fetch failed for {rel}: {e}')
    return None


# ── Embedder loader (mirror icon_matcher._get_ml_session for embedder kind) ──

def _load_embedder(model_dir: Path):
    """Returns (model, gallery_emb, gallery_lbl, label_map)."""
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import efficientnet_b0

    label_map = {
        int(k): v for k, v in json.loads(
            (model_dir / 'embedder_label_map.json').read_text(encoding='utf-8')
        ).items()
    }
    gallery = np.load(str(model_dir / 'embedding_index.npz'))
    gallery_emb = gallery['embeddings'].astype(np.float32)
    gallery_lbl = gallery['labels'].astype(np.int32)
    embed_dim = int(gallery_emb.shape[1])

    backbone = efficientnet_b0(weights=None)
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()

    class Embedder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.proj = nn.Linear(in_features, embed_dim)

        def forward(self, x):
            f = self.backbone(x)
            e = self.proj(f)
            return F.normalize(e, dim=1)

    model = Embedder()
    state = torch.load(str(model_dir / 'icon_embedder.pt'), map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model, gallery_emb, gallery_lbl, label_map


def _classify_png(png_bytes: bytes, model, gallery_emb, gallery_lbl, label_map):
    """Decode PNG → run embedder → return (label, cos_sim)."""
    import cv2
    import numpy as np
    import torch

    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return '', 0.0
    rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    inp = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = (inp - mean) / std
    inp = np.expand_dims(np.transpose(inp, (2, 0, 1)), axis=0)
    t = torch.from_numpy(inp)
    with torch.no_grad():
        emb = model(t).numpy()[0]
    sims = gallery_emb @ emb
    top = int(np.argmax(sims))
    return label_map.get(int(gallery_lbl[top]), ''), float(max(0.0, min(1.0, sims[top])))


# ── HF write ──────────────────────────────────────────────────────────────────

def _save_cleaned(envelope: dict, cleaned: dict[str, str]) -> bool:
    """Write knowledge.json back to HF, preserving the envelope fields
    (processed_contributions, watermark_date, etc.) so admin_merge.py
    doesn't lose state."""
    from huggingface_hub import HfApi
    payload_obj = dict(envelope)
    payload_obj['knowledge']  = cleaned
    payload_obj['entries']    = len(cleaned)
    payload_obj['updated_at'] = datetime.now(UTC).isoformat() + 'Z'
    payload = json.dumps(payload_obj, ensure_ascii=False, indent=2).encode('utf-8')
    api = HfApi(token=HF_TOKEN)
    try:
        api.upload_file(
            path_or_fileobj=payload,
            path_in_repo='knowledge.json',
            repo_id=HF_REPO_ID,
            repo_type='dataset',
            commit_message=f'admin_scrub_knowledge: {len(cleaned)} entries '
                           f'({datetime.now(UTC).strftime("%Y-%m-%d %H:%M")} UTC)',
        )
        return True
    except Exception as e:
        log.error(f'upload knowledge.json failed: {e}')
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Scrub poisoned (blank-icon → real-name) entries from '
                    'community knowledge.json using the bootstrapped embedder.',
    )
    parser.add_argument('--apply', action='store_true',
                        help='Write cleaned knowledge.json back to HF '
                             '(default: dry-run).')
    parser.add_argument('--limit', type=int, default=0,
                        help='Inspect at most N entries (0 = all). Useful '
                             'for sampling before a full run.')
    parser.add_argument('--virtual-conf', type=float, default=VIRTUAL_CONF,
                        help=f'Min cos-sim for embedder verdict to drop an '
                             f'entry (default {VIRTUAL_CONF}).')
    args = parser.parse_args()

    _require_hf()

    print('=' * 64)
    print('WARP Knowledge Scrub')
    print(f'Repo:    {HF_REPO_ID}')
    print(f'Mode:    {"APPLY" if args.apply else "DRY-RUN"}')
    print(f'Cutoff:  embedder virtual conf >= {args.virtual_conf}')
    print('=' * 64)

    print('\nLoading knowledge.json...')
    envelope  = _download_knowledge()
    knowledge = envelope.get('knowledge', envelope) \
        if isinstance(envelope, dict) else {}
    if not isinstance(knowledge, dict):
        print('ERROR: knowledge.json has no `knowledge` field', file=sys.stderr)
        sys.exit(2)
    print(f'  {len(knowledge)} entries')

    print('\nCloning contributions tree (shallow)...')
    snap = _clone_contributions()
    print('  building phash → contributions index...')
    phash_idx = _build_phash_index(snap)
    print(f'  {len(phash_idx)} distinct pHashes with surviving contributions')

    print('\nDownloading embedder assets...')
    model_dir = Path('/tmp/warp-scrub-model')
    model_dir.mkdir(exist_ok=True)
    _download_embedder_assets(model_dir)
    print('  loading embedder...')
    model, gallery_emb, gallery_lbl, label_map = _load_embedder(model_dir)
    print(f'  embedder loaded ({len(label_map)} classes, '
          f'gallery={len(gallery_lbl)})')

    items = list(knowledge.items())
    if args.limit > 0:
        items = items[:args.limit]
        print(f'\n[--limit] Inspecting first {len(items)} entries')

    print(f'\nScanning {len(items)} entries...')
    cleaned: dict[str, str] = {}
    poisoned: list[tuple[str, str, str, float]] = []  # phash, name, vlabel, vconf
    no_evidence: int = 0
    inspected: int = 0

    for ph, name in items:
        contribs = phash_idx.get(ph, [])
        if not contribs:
            # Can't verify — preserve the entry. This is conservative;
            # contributions may have been compacted out by watermark
            # advancement in admin_merge.py.
            cleaned[ph] = name
            no_evidence += 1
            continue

        png = _fetch_contribution_png(contribs[0])
        if not png:
            cleaned[ph] = name
            no_evidence += 1
            continue

        inspected += 1
        vlabel, vconf = _classify_png(png, model, gallery_emb, gallery_lbl, label_map)
        if vlabel in VIRTUAL_NAMES and vconf >= args.virtual_conf:
            poisoned.append((ph, name, vlabel, vconf))
            # Drop the entry — do NOT add to cleaned.
        else:
            cleaned[ph] = name

        if inspected % 100 == 0:
            print(f'  ... {inspected} crops classified, '
                  f'{len(poisoned)} poison so far')

    # ── Report ────────────────────────────────────────────────────────────────
    print('\n── Scan complete ──')
    print(f'  Inspected (had contribution crop): {inspected}')
    print(f'  Skipped (no surviving evidence):   {no_evidence}')
    print(f'  Poisoned (would drop):             {len(poisoned)}')
    print(f'  Surviving entries:                 {len(cleaned)} '
          f'(was {len(knowledge)})')

    if poisoned:
        print('\nPoison samples (first 20):')
        by_name: dict[str, int] = defaultdict(int)
        for _, n, _, _ in poisoned:
            by_name[n] += 1
        for n in sorted(by_name, key=by_name.get, reverse=True)[:20]:
            print(f'  {by_name[n]:4d} × {n!r}')

    # ── Apply ─────────────────────────────────────────────────────────────────
    if not args.apply:
        print('\nDry-run finished. Re-run with --apply to write back to HF.')
        return
    if not poisoned:
        print('\nNothing to scrub — knowledge.json is clean.')
        return
    print(f'\nUploading cleaned knowledge.json '
          f'({len(cleaned)} entries, {len(poisoned)} dropped)...')
    if _save_cleaned(envelope, cleaned):
        print('OK — knowledge.json updated on HF.')
    else:
        print('FAILED — knowledge.json was NOT updated.', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
