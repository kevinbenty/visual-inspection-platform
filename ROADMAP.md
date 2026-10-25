# Roadmap (from the project plan)

- [x] Week 1 – Scope: Option B (anomaly localisation) on MVTec AD; split policy defined
- [x] Week 2 – Baseline pipeline code (global kNN)  · [ ] run and record results
- [x] Week 3 – PatchCore code  · [ ] compare on ≥3 categories, inspect failures
- [x] Week 4 – FastAPI endpoint with validation and error handling
- [x] Week 5 – Streamlit upload, heatmap, history
- [x] Week 6 – Docker, pytest, config, logging  · [ ] CI (GitHub Actions)
- [ ] Week 7 – Benchmark CPU vs GPU, coreset ratio vs latency/AUROC, lighting/brightness robustness
- [ ] Week 8 – Fill results tables, architecture diagram, demo video, short technical report

Extensions that fit an edge-AI profile: ONNX export of the backbone, INT8
quantisation, FAISS for the kNN search, and a latency/accuracy trade-off curve
over coreset ratio.
