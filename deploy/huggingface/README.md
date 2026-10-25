---
title: Visual Inspection Platform
emoji: 🔍
colorFrom: gray
colorTo: indigo
sdk: docker
app_port: 7860
license: cc-by-nc-sa-4.0
short_description: Spot and highlight defects in industrial part images
---

# Visual Inspection Platform

Upload a photo of a part, or pick a sample, to see whether the model flags it and where the
unusual region is. The model learned only from images of normal parts (PatchCore-style
anomaly detection), so it needs no labelled defects.

Source code, method and evaluation: https://github.com/kevinbenty/visual-inspection-platform

**Dataset.** The model and sample images come from MVTec AD (Bergmann et al., CVPR 2019 /
IJCV 2021), licensed CC BY-NC-SA 4.0. This demo is non-commercial.

**Limits.** A portfolio prototype, not a production inspection system. It only knows the
category it was trained on, and its score is not a calibrated probability.
