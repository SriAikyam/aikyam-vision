# aikyam-vision — Service Context

Parent context (infra, service map, Redis namespaces, kubectl shortcuts):
→ `/Users/sk040d/myprojects/narendra/prod-setup/CLAUDE.md` (auto-loaded)

---

## What This Service Does

Extracts visual/audio signals from post media (image + video) and publishes cluster scores
to Kafka. SimClusters picks up those scores and merges them into each post's embedding vector.

Image: `ghcr.io/sriaikyam/aikyam-vision`  
Language: Python 3.11 / FastAPI  
Two modes from the same image, selected by `MODE` env var:
- `MODE=api` — FastAPI HTTP server, port 8000, endpoints: `/analyze/image`, `/analyze/video`, `/analyze/text`
- `MODE=worker` — Kafka consumer loop, no HTTP port

---

## Position in the Pipeline

```
Kafka: aikyam.post.created
    └──► aikyam-vision (MODE=worker)
              downloads media URL
              runs: CLIP + keywords + EXIF + GPS + pHash
              → publishes Kafka: aikyam.vision.scores
                                    └──► aikyam-simclusters VisionScoreConsumer
                                              merges 0.6×vision + 0.4×existing
```

Vision NEVER HTTP-calls SimClusters. Fully decoupled via Kafka.

---

## File Map

| File | Purpose |
|------|---------|
| `vision/config.py` | All env vars with defaults |
| `vision/cluster_mapper.py` | Combines all signals → `VisionResult(clusters, confidence, phash, sources)` |
| `vision/clip_model.py` | Lazy CLIP, zero-shot against 19 cluster prompt sets |
| `vision/whisper_model.py` | Lazy faster-whisper (tiny, int8), VAD filter, multilingual |
| `vision/exif_extractor.py` | PIL EXIF fallback — GPS decode + sacred geography bbox (14 sites) |
| `vision/exiftool_extractor.py` | PyExifTool — full EXIF, JPEG quant tables, AI-generated detection |
| `vision/phash.py` | pHash + dHash + aHash (imagehash) |
| `vision/file_info.py` | SHA-256, MIME, dimensions, file size |
| `vision/video_processor.py` | ffprobe full stream metadata, GPS, device tags, 5 frames |
| `worker/consumer.py` | Kafka consumer: download → analyze → publish vision scores |
| `api/routes.py` | FastAPI endpoints |
| `main.py` | Entry point: `python main.py api` or `python main.py worker` |

---

## Kafka Topics

| Topic | Direction | Group |
|-------|-----------|-------|
| `aikyam.post.created` | consume | `aikyam-vision-v1` |
| `aikyam.vision.scores` | publish | — |

### Vision score message format (published to aikyam.vision.scores)

```json
{
  "postId":     "abc123",
  "clusters":   {"cluster_01": 0.9, "cluster_53": 0.7},
  "confidence": 0.85,
  "phash":      "cdf39c33634c4949",
  "sources":    ["keywords", "clip"]
}
```

---

## What Works Without ML (no model downloads)

- Text keyword scoring → cluster scores (19-cluster keyword map)
- pHash + dHash + aHash (imagehash library)
- Full EXIF via PyExifTool — camera, lens, GPS, timestamps
- GPS → sacred geography cluster (14 pilgrimage site bounding boxes)
- JPEG quantization tables + quality estimate
- AI-generated detection (C2PA, Software field signatures, round dimensions)
- Video: container, codec, resolution, fps, color, audio, device tags, GPS

## What Needs Env Vars (model downloads)

| Feature | Env var | Download size |
|---------|---------|--------------|
| CLIP image scoring | `AIKYAM_CLIP_MODEL_ENABLED=true` | ~150MB |
| Whisper audio | `AIKYAM_WHISPER_MODEL_ENABLED=true` | ~40MB (tiny) |

Both also need `INSTALL_ML=true` in the `release-image` GitHub Actions workflow.

---

## Payload Parsing (Kafka event field names)

The `aikyam.post.created` event has this structure — easy to get wrong:

```python
nested    = payload.get("payload", {}) or {}
caption   = nested.get("postText", "") or payload.get("text", "") or ""
image_url = nested.get("postImageUrl")          # NOT payload["imageUrl"]
entity_id = payload.get("templeId") or payload.get("entityId")
post_id   = payload.get("postId", "") or payload.get("entityId", "")
```

---

## Quick Smoke Test (API mode)

```bash
# Start API locally
docker run -d -p 8010:8000 ghcr.io/sriaikyam/aikyam-vision:dev-latest

# Health
curl http://localhost:8010/health

# Text scoring (no model needed)
curl -s -X POST http://localhost:8010/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"post_id":"t1","caption":"Om Namah Shivaya Kedarnath #mahadev"}' | jq .clusters

# Image scoring (use picsum.photos — Wikipedia blocks urllib)
curl -s -X POST http://localhost:8010/analyze/image \
  -H "Content-Type: application/json" \
  -d '{"post_id":"t2","image_url":"https://picsum.photos/seed/temple/800/600","caption":"puja aarti mandir"}' | jq '{clusters,phash,ai_generated}'
```

---

## Helm Chart (deploy via Aikyam-platform)

Two deployments in one chart: `aikyam-vision-api` and `aikyam-vision-worker`.

```bash
KC="kubectl --kubeconfig=/Users/sk040d/myprojects/narendra/prod-setup/ovh-kube/kubeconfig-pol47r.yml -n aikyam-dev"

# Apply secret first
$KC apply -f ../Aikyam-platform/kubernetes/secrets/vision/vision-secret-dev.yaml

# Deploy
helm upgrade --install aikyam-vision \
  ../Aikyam-platform/aikyam-vision/helm \
  -f ../Aikyam-platform/aikyam-vision/helm/values-dev.yaml \
  --kubeconfig ../ovh-kube/kubeconfig-pol47r.yml \
  -n aikyam-dev
```

---

## Build & Deploy

```bash
# Build image without ML (keywords + EXIF only)
gh workflow run release-image.yml --repo SriAikyam/aikyam-vision -f image_tag=dev-latest

# Build image WITH CLIP + Whisper
gh workflow run release-image.yml --repo SriAikyam/aikyam-vision \
  -f image_tag=dev-latest -f install_ml=true

# Restart worker after new image
KC="kubectl --kubeconfig=/Users/sk040d/myprojects/narendra/prod-setup/ovh-kube/kubeconfig-pol47r.yml -n aikyam-dev"
$KC rollout restart deployment/aikyam-vision-worker
$KC rollout restart deployment/aikyam-vision-api
```

---

## AI-Generated Image Detection Logic

Flags `ai_generated: true` if ≥2 signals OR any direct metadata signature:
1. EXIF Software/CreatorTool contains: midjourney, dall-e, stable diffusion, comfyui, firefly, etc.
2. C2PA manifest present in XMP
3. No camera make/model AND metadata not stripped (genuine photos always have camera info)
4. Image dimensions match common AI sizes: 512×512, 1024×1024, 768×512, 1344×768, etc.
