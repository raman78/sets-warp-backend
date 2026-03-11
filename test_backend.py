#!/usr/bin/env python3
# test_backend.py
# Uruchom lokalnie: python test_backend.py <ścieżka_do_screenshota>
#
# Testuje:
#   1. GET  /health
#   2. POST /contribute  (wycina crop ze screenshota i wysyła)
#   3. GET  /knowledge   (sprawdza czy backend zwraca knowledge base)

import sys
import base64
import json
import urllib.request
import urllib.error

BACKEND_URL = 'https://sets-warp-backend.onrender.com'


def test_health():
    print("1. GET /health ...")
    req = urllib.request.Request(f'{BACKEND_URL}/health')
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
    print(f"   ✓ {data}")


def test_contribute(screenshot_path: str):
    import cv2
    print(f"2. POST /contribute (crop z {screenshot_path}) ...")

    img = cv2.imread(screenshot_path)
    if img is None:
        print(f"   ✗ Nie można otworzyć pliku")
        return
    h, w = img.shape[:2]
    print(f"   Screenshot: {w}x{h}")

    # Wytnij środek obrazu jako crop testowy
    cx, cy = w // 2, h // 2
    crop = img[cy-25:cy+25, cx-25:cx+25]
    ok, buf = cv2.imencode('.png', crop)
    crop_b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    print(f"   Crop: {crop.shape}, PNG: {len(buf)} bytes")

    payload = json.dumps({
        'install_id':    'test-install-001',
        'phash':         'a1b2c3d4e5f60001',
        'crop_png_b64':  crop_b64,
        'item_name':     'Test Item Name',
        'wrong_name':    '',
        'confirmed':     True,
        'warp_version':  '0.4.0',
        'timestamp':     '2026-03-11T13:00:00Z',
    }).encode('utf-8')

    req = urllib.request.Request(
        f'{BACKEND_URL}/contribute',
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    print(f"   ✓ {data}")


def test_knowledge():
    print("3. GET /knowledge ...")
    req = urllib.request.Request(f'{BACKEND_URL}/knowledge')
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
    count = len(data.get('knowledge', {}))
    print(f"   ✓ {count} entries in knowledge base")


if __name__ == '__main__':
    screenshot = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        test_health()
    except Exception as e:
        print(f"   ✗ {e}")

    if screenshot:
        try:
            test_contribute(screenshot)
        except urllib.error.HTTPError as e:
            print(f"   ✗ HTTP {e.code}: {e.read().decode()}")
        except Exception as e:
            print(f"   ✗ {e}")
    else:
        print("2. POST /contribute — pomiń (brak screenshota, podaj jako argument)")

    try:
        test_knowledge()
    except Exception as e:
        print(f"   ✗ {e}")
