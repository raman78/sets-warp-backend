#!/usr/bin/env python3
# test_backend.py
# Run locally: python test_backend.py <path_to_screenshot>
#
# Tests:
#   1. GET  /health
#   2. GET  /model/version
#   3. POST /contribute  (extracts crop from screenshot and sends it)
#   4. GET  /knowledge   (checks if backend returns knowledge base)

import sys
import base64
import json
import urllib.request
import urllib.error
import time

BACKEND_URL = 'https://sets-warp-backend.onrender.com'
GLOBAL_TIMEOUT = 60  # Increased for Render cold starts


def test_health():
    print("1. GET /health ...")
    req = urllib.request.Request(f'{BACKEND_URL}/health')
    with urllib.request.urlopen(req, timeout=GLOBAL_TIMEOUT) as r:
        data = json.loads(r.read())
    print(f"   ✓ {data}")


def test_model_version():
    print("2. GET /model/version ...")
    req = urllib.request.Request(f'{BACKEND_URL}/model/version')
    with urllib.request.urlopen(req, timeout=GLOBAL_TIMEOUT) as r:
        data = json.loads(r.read())
    if data.get('available'):
        print(f"   ✓ Model available: version={data.get('version')} trained_at={data.get('trained_at')}")
    else:
        print("   ✓ No model published yet.")


def test_contribute(screenshot_path: str):
    import cv2
    import numpy as np
    print(f"3. POST /contribute (crop from {screenshot_path}) ...")

    img = cv2.imread(screenshot_path)
    if img is None:
        print(f"   ✗ Cannot open file")
        return
    h, w = img.shape[:2]
    print(f"   Screenshot: {w}x{h}")

    # Extract center of the image as a test crop
    cx, cy = w // 2, h // 2
    crop = img[cy-32:cy+32, cx-32:cx+32]
    ok, buf = cv2.imencode('.png', crop)
    crop_b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    print(f"   Crop: {crop.shape}, PNG: {len(buf)} bytes")

    payload = json.dumps({
        'install_id':    'test-agent-cli',
        'phash':         'abcdef1234567890',
        'crop_png_b64':  crop_b64,
        'item_name':     'CLI Test Item',
        'wrong_name':    '',
        'confirmed':     True,
        'warp_version':  '0.5.0',
        'timestamp':     time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }).encode('utf-8')

    req = urllib.request.Request(
        f'{BACKEND_URL}/contribute',
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=GLOBAL_TIMEOUT) as r:
        data = json.loads(r.read())
    print(f"   ✓ {data}")


def test_knowledge():
    print("4. GET /knowledge ...")
    req = urllib.request.Request(f'{BACKEND_URL}/knowledge')
    with urllib.request.urlopen(req, timeout=GLOBAL_TIMEOUT) as r:
        data = json.loads(r.read())
    
    knowledge = data.get('knowledge', {})
    count = len(knowledge)
    print(f"   ✓ {count} entries in knowledge base")
    if count > 0:
        samples = list(knowledge.items())[:3]
        for ph, name in samples:
            print(f"     - {ph}: {name}")


if __name__ == '__main__':
    screenshot = sys.argv[1] if len(sys.argv) > 1 else None

    print(f"Starting backend tests (timeout={GLOBAL_TIMEOUT}s) for: {BACKEND_URL}")
    print("-" * 50)

    try:
        test_health()
    except Exception as e:
        print(f"   ✗ Health check failed: {e}")

    try:
        test_model_version()
    except Exception as e:
        print(f"   ✗ Model version check failed: {e}")

    if screenshot:
        try:
            test_contribute(screenshot)
        except urllib.error.HTTPError as e:
            content = e.read().decode()
            print(f"   ✗ HTTP {e.code}: {content}")
        except Exception as e:
            print(f"   ✗ Contribution failed: {e}")
    else:
        print("3. POST /contribute — skipped (no screenshot provided)")

    try:
        test_knowledge()
    except Exception as e:
        print(f"   ✗ Knowledge base check failed: {e}")

    print("-" * 50)
    print("Test run complete.")
