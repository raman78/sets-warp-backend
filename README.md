# SETS WARP Backend

Backend service for the **Weapon & Armor Recognition Program (WARP)**. It handles community-contributed data (icon crops and screenshots), performs democratic voting, and automates model training.

## 🚀 System Architecture
- **FastAPI (Render)**: Receives contributions and serves the knowledge base/models.
- **Hugging Face Hub**: Stores raw data (Dataset) and hosted models.
- **GitHub Actions**: Automates model training (EfficientNet-B0 and MobileNetV3).

---

## 🛠 Setup & Configuration

### 1. Hugging Face Setup
- Create a **Dataset** repository (e.g., `sets-sto/warp-knowledge`).
- Generate a **Write Token** in HF Settings → Access Tokens.

### 2. GitHub Actions Setup
- Go to your GitHub repository **Settings → Secrets and variables → Actions**.
- Add the following **Secrets**:
  - `HF_TOKEN`: Your Hugging Face write token.
- (Optional) Add **Variables** if you use non-default repo names:
  - `HF_DATASET`: Source dataset repo.
  - `HF_REPO_ID`: Target model repo.

### 3. Render Deployment (Production)
Deploy the FastAPI service to Render and set these **Environment Variables**:
- `HF_TOKEN`: Your HF write token.
- `HF_REPO_ID`: Your model/data repo (e.g., `sets-sto/warp-knowledge`).
- `ADMIN_KEY`: A secret string to protect admin endpoints.
- `GH_TOKEN`: GitHub Personal Access Token (with `workflow` scope) to trigger training.
- `GH_REPO`: Your GitHub repository path (e.g., `username/repo`).

---

## 🛰 Endpoints
- `GET /health`: Service status.
- `GET /knowledge`: Returns the merged icon knowledge base (phash -> name).
- `GET /model/version`: Metadata for the latest trained model.
- `POST /contribute`: Accepts crop PNG + metadata from clients.
- `POST /webhooks/hf-dataset`: Triggered by HF when the dataset is updated (starts GitHub training).
- `POST /admin/merge`: (Requires `X-Admin-Key`) Merges raw contributions into `knowledge.json`.

---

## 🧠 Model Training
Training runs automatically every hour via GitHub Actions, or can be triggered manually:

### Local Training
```sh
# Requires PyTorch, Torchvision, OpenCV
python admin_train.py --train --min 1
```

### Manual Merge
```sh
python admin_merge.py --apply
```

---

## 📄 Documentation
- Detailed guidelines: [CLAUDE.md](./CLAUDE.md)
- Technical Overview: [docs/technical_overview.md](./docs/technical_overview.md)
- User Guide: [docs/user_guide.md](./docs/user_guide.md)
- Recent Changes: [CHANGELOG.md](./CHANGELOG.md)

## ⚖ License
MIT
