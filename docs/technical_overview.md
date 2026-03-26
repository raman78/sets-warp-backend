# Technical Overview - SETS WARP Backend

## System Architecture
The backend serves as a central hub for the WARP (Weapon & Armor Recognition Program) ecosystem. It facilitates community-driven data collection and automated model training.

### Components
1. **FastAPI Service (`main.py`)**: 
   - Handles contributions from mobile clients.
   - Serves the merged knowledge base and model version metadata.
   - Manages rate limiting and basic data validation.
2. **Model Trainer (`admin_train.py`)**:
   - A standalone script for democratic voting on contributions.
   - Trains two PyTorch models: `icon_classifier` (EfficientNet-B0) and `screen_classifier` (MobileNetV3-Small).
   - Uploads trained models to Hugging Face.
3. **Data Storage (Hugging Face Dataset)**:
   - Stores raw contributions, crops, and screenshots.
   - Hosts the final models and metadata.

## Core Workflows

### 1. Data Contribution
- Client sends a base64 encoded PNG crop + metadata to `/contribute`.
- Backend validates the image (size, uniformity) and uploads it to the `contributions/` folder in the HF Dataset.

### 2. Model Training (Automated/Manual)
- Triggered via CLI or Webhook.
- `admin_train.py` scans `staging/` folders in the dataset.
- Applies democratic voting: 1 unique `install_id` = 1 vote.
- Downloads winning samples and trains/fine-tunes models.
- Uploads updated models and `model_version.json`.

### 3. Knowledge Merging
- `admin_merge.py` (or `/admin/merge` endpoint) consolidates raw contributions into a unified `knowledge.json` mapping.

## Infrastructure
- **Hosting**: Render (FastAPI service).
- **CI/CD**: Bitbucket Pipelines (for automated training triggers).
- **Storage**: Hugging Face Hub (Dataset repo).
