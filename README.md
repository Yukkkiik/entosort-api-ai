# EntoSort Inference API

FastAPI service untuk inference model LarvaSort Vision (2 kelas: `larva_active`, `prepupa_pupa`).
Menyediakan 2 endpoint terpisah supaya kamu bisa bandingkan performa:

- `POST /predict/onnx` → pakai `model_int8.onnx` (lebih ringan & cepat di CPU)
- `POST /predict/pt` → pakai `best.pt` (via ultralytics, lebih berat karena narik torch)

## Struktur project

```
entosort-api/
├── main.py           # FastAPI app
├── requirements.txt
├── Dockerfile
├── railway.json
├── .dockerignore
└── models/
    ├── best.pt
    └── model_int8.onnx
```

## Deploy ke Railway

### Opsi A — via Railway CLI (paling gampang)

1. Install CLI: `npm i -g @railway/cli`
2. Login: `railway login`
3. Di dalam folder `entosort-api/`:
   ```
   railway init
   railway up
   ```
4. Setelah deploy selesai, generate domain publik:
   ```
   railway domain
   ```

### Opsi B — via GitHub

1. Push folder ini ke repo GitHub baru.
2. Di Railway dashboard → New Project → Deploy from GitHub repo.
3. Railway otomatis detect `Dockerfile` dan build.
4. Buka tab Settings → Networking → Generate Domain.

> Catatan: karena `best.pt` butuh ultralytics + torch, image Docker akan cukup besar (~2GB+) dan cold start lebih lambat dibanding endpoint ONNX. Kalau nanti sudah fix pakai salah satu model saja, hapus dependency yang tidak dipakai (`ultralytics` di requirements.txt) supaya build lebih cepat dan hemat resource/biaya di Railway.

## Cara test endpoint

### Encode gambar ke base64 (contoh Python)

```python
import base64
with open("contoh.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
print(b64[:50], "...")
```

### Request contoh (curl)

```bash
curl -X POST https://<domain-railway-kamu>/predict/onnx \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<isi base64 di sini>", "conf": 0.25}'
```

### Contoh response

```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "larva_active",
      "confidence": 0.87,
      "bbox": [120.5, 45.2, 310.8, 290.1]
    }
  ],
  "inference_time_ms": 42.31,
  "image_width": 1280,
  "image_height": 720
}
```

`bbox` dalam format `[x1, y1, x2, y2]` piksel, relatif ke ukuran gambar asli yang kamu kirim (bukan 640x640 hasil resize internal).

## Testing lokal sebelum deploy

```bash
pip install -r requirements.txt
uvicorn main:app --reload
# buka http://localhost:8000/docs untuk Swagger UI interaktif
```

## Environment variable

Railway otomatis inject `PORT` — sudah dihandle di `Dockerfile` (`--port ${PORT:-8000}`), tidak perlu setting manual.
