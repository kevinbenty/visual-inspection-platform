# Visual Inspection Platform

Unsupervised anomaly detection and localisation for industrial images, served
through a tested FastAPI service with a Streamlit review dashboard.

**Problem.** On a production line most parts are fine, and defects are rare and
varied, so collecting labelled examples of every defect is impractical. This
project learns what *normal* parts look like and flags images that deviate,
highlighting the region responsible.

**Intended users.** Quality engineers evaluating whether learned visual
inspection is worth piloting on a line, and reviewers who need to see *where*
the model thinks the problem is.

**Not intended for:** safety-critical pass/fail decisions, estimating defect
severity, or categories other than those it was trained on. See
[MODEL_CARD.md](MODEL_CARD.md).

## How it works

```
image ─► resize/crop ─► frozen WideResNet-50 (layer2+layer3) ─► patch features
                                                                  │
            memory bank of NORMAL patch features (coreset) ◄──────┘ (training)
                                                                  │
   nearest-neighbour distance per patch ─► anomaly map ─► max = image score
                                                                  │
                         score ≥ threshold ─► "flagged for review"
```

* **Main method:** PatchCore-style patch memory bank with greedy coreset
  subsampling (Roth et al., *Towards Total Recall in Industrial Anomaly
  Detection*, CVPR 2022). Simplified re-implementation, not the official code.
* **Baseline:** kNN on global image embeddings (image-level only, no heatmap).
* **Threshold:** the 99th percentile of scores on held-out *normal training*
  images. No test images and no defect labels are used to choose it.

## Dataset

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), 15
categories of industrial objects and textures. Licensed **CC BY-NC-SA 4.0**
(non-commercial; cite Bergmann et al., CVPR 2019 / IJCV 2021). Check the
current terms on the MVTec page before use. The data is not redistributed here.

Download it from the MVTec page and extract so that you have
`data/mvtec_ad/bottle/train/good/*.png`.

| Split | Source | Used for |
|---|---|---|
| fit | `train/good` minus calibration subset (seeded) | memory bank |
| calibration | 10% of `train/good`, seed 42 | decision threshold only |
| test | official `test/` (normal + defect types) | final metrics only |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make install
make test                     # unit + API tests, no data needed
make train CAT=bottle         # build memory bank, calibrate threshold
make evaluate CAT=bottle      # metrics -> reports/patchcore/bottle/metrics.json
make baseline CAT=bottle      # global-kNN baseline for comparison
make bench CAT=bottle         # latency / memory
make api CAT=bottle           # http://localhost:8000/docs
make dashboard                # http://localhost:8501 (second terminal)
```

Docker (after training, so `models/` exists):

```bash
docker compose up --build
```

### API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/inspect` | multipart image → score, threshold, verdict, heatmap (base64 PNG) |
| `GET` | `/history?limit=50&only_anomalous=false` | past inspections |
| `GET` | `/health` | liveness and model status |

Rejects non-image content types (415), corrupt files (422), files over 10 MB (413).

## Results

> Fill in from `reports/` after running. Report only what you measured.

| Category | Method | Image AUROC | Pixel AUROC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| bottle | global kNN (ResNet-18) | – | n/a | – | – | – |
| bottle | PatchCore (WRN-50, 10% coreset) | – | – | – | – | – |

Latency (`make bench`): hardware ___, mean ___ ms, p95 ___ ms, memory bank ___ MB.

## Repository layout

```
app/        FastAPI service, inference wrapper, SQLite history
dashboard/  Streamlit UI (talks to the API only)
src/        data indexing, preprocessing, models, train/evaluate/benchmark
tests/      preprocessing, scoring core, split, API (fake model)
configs/    experiment config (YAML, overridable with --set)
reports/    metrics.json + failure-case images per run
```

## Production readiness

This is a portfolio prototype. It has input validation, tests, logging,
containerisation and a documented evaluation. It does **not** have
authentication, rate limiting, drift monitoring, model registry/versioning
beyond a name string, or validation on real line images.
