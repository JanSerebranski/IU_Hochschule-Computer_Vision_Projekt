import requests

API_URL = "http://127.0.0.1:5000"

TEST_IMAGES = {
    "valid": "testdata/face.jpg",           # Ein valides Bild mit Gesicht
    "no_face": "testdata/no_face.jpg",      # Ein Bild ohne Gesicht
    "invalid": "testdata/invalid.txt",      # Keine Bilddatei
    "large": "testdata/large_image.jpg"     # Sehr großes Bild (optional)
}

def test_analyze():
    print("\n--- /analyze Tests ---")
    for name, path in TEST_IMAGES.items():
        files = {"image": open(path, "rb")}
        try:
            response = requests.post(f"{API_URL}/analyze", files=files)
            data = response.json()
            print(f"Test: {name:8} | success: {data.get('success')} | error: {data.get('error')}")
        except Exception as e:
            print(f"Test: {name:8} | Fehler: {e}")

def test_batch_analyze():
    print("\n--- /batch-analyze Tests ---")
    files = [("images", open(TEST_IMAGES["valid"], "rb")),
             ("images", open(TEST_IMAGES["no_face"], "rb")),
             ("images", open(TEST_IMAGES["invalid"], "rb"))]
    try:
        response = requests.post(f"{API_URL}/batch-analyze", files=files)
        data = response.json()
        for res in data.get("results", []):
            print(f"Batch: {res.get('filename', '?'):12} | error: {res.get('error')}")
    except Exception as e:
        print(f"Batch-Test Fehler: {e}")

if __name__ == "__main__":
    test_analyze()
    test_batch_analyze() 