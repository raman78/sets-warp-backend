# User Guide - SETS WARP Backend

## Overview
The backend provides essential services for the WARP application, including data contribution, knowledge updates, and access to the latest centrally-trained models.

## Client-Side Integration

### 1. Contributing Data
Clients can contribute to the community knowledge base by sending data to the `/contribute` endpoint. 
- **Endpoint**: `POST /contribute`
- **Request Body**:
  - `install_id`: Unique identifier for the client installation.
  - `phash`: Perceptual hash of the icon crop.
  - `crop_png_b64`: Base64-encoded PNG of the icon crop.
  - `item_name`: Confirmed name of the item.
  - `warp_version`: Version of the client application.

### 2. Fetching Knowledge
Clients can retrieve the latest merged knowledge base to improve local recognition.
- **Endpoint**: `GET /knowledge`
- **Response**: A JSON object mapping perceptual hashes to item names.

### 3. Checking for Model Updates
The client can check if a new centrally-trained model is available for download.
- **Endpoint**: `GET /model/version`
- **Response**: Contains `available: true` and a `version` (hash) if an update exists.

## Admin Procedures

### 1. Merging Contributions
Confirmed contributions can be merged into the master `knowledge.json` using the `/admin/merge` endpoint.
- **Requirement**: `X-Admin-Key` header with the secret `ADMIN_KEY`.

### 2. Manual Model Training
Training can be manually triggered by running `admin_train.py` with appropriate flags:
- `python admin_train.py --train --min 1`

## Support and Troubleshooting
- **Rate Limiting**: Users are limited to a specific number of requests per day to prevent abuse.
- **Health Check**: Verify if the backend is online at `/health`.
