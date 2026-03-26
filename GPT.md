# SETS WARP Backend - Agent Guidelines

## Project Overview
Backend service for WARP (Weapon & Armor Recognition Program). 
Handles democratic voting on community data and automated model training (Icon Classifier & Screen Classifier) using PyTorch.

## Tech Stack
- Python 3.11+
- PyTorch & Torchvision (Model training)
- OpenCV (Image processing)
- Hugging Face Hub (Storage & Model hosting)
- Render (Production deployment)

## Strict Rules
1. **Language**: All code comments, logs, and commit messages MUST be in English.
2. **Environment**: Code must be tested locally and compatible with Render (production).
3. **Communication**: Verify bidirectional communication (client-server and server-HF).
4. **Data Integrity**: Always verify that data/crops are correctly uploaded to the HF Dataset.
5. **Model Training**: Verify that training starts correctly and models are valid before upload.
6. **Error Handling**: Strict verification of connectivity and HF timeouts. Use non-recursive listing.
7. **Security**: NEVER hardcode tokens (HF_TOKEN). Use environment variables or .env files.
8. **Atomic Uploads**: Group model files into a single HF commit using `CommitOperationAdd`.
9. **Documentation**: Actively search for and maintain all project documentation.
   - Update `CHANGELOG.md` in the root directory for all changes.
   - Maintain technical and user documentation in the `/docs` folder (create if missing).
   - Update `README.md` and any other `.md` files to reflect architectural or usage changes.
   - Ensure all documentation is consistent and cross-linked.

## Core Commands
- **Dry run training**: `python admin_train.py`
- **Real training + upload**: `python admin_train.py --train --min 1`
- **Skip unchanged**: `python admin_train.py --train --skip-if-unchanged`
- **Check Backend**: `python main.py`

## Development Best Practices
- **Type Hints**: Use Python type hints everywhere.
- **Logging**: Use `log = logging.getLogger(__name__)`.
- **Resource Management**: Be mindful of RAM limits on Render (adjust BATCH_SIZE if needed).
- **Style**: Match the existing script style (flat structure, clear section separators).

## HF Configuration
- `HF_TOKEN`: Required for write access.
- `HF_DATASET`: Source dataset (default: `sets-sto/sto-icon-dataset`).
- `HF_REPO_ID`: Target model repo (default: `sets-sto/warp-knowledge`).
