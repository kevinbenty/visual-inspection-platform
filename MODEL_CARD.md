# Model card: PatchCore-style anomaly detector

**Model.** Frozen ImageNet WideResNet-50-2 features (layer2 + layer3, 3×3 local
average), greedy-coreset memory bank of normal patches, 1-NN patch distance.
One model per MVTec AD category.

**Intended use.** Research and demonstration of unsupervised visual anomaly
detection; ranking images for human review.

**Out of scope.** Safety or certification decisions; defect severity; defect
type classification; categories, cameras or lighting not seen in training.

**Training data.** MVTec AD `train/good` for one category (normal images only).
**Threshold.** 99th percentile of scores on a seeded 10% hold-out of
`train/good`. Expect roughly 1% false alarms on in-distribution normals;
the true rate on test is reported in `metrics.json`.

**Evaluation data.** Official MVTec AD test split for that category.

**Metrics.** Image AUROC, pixel AUROC, precision / recall / F1 / confusion at
the calibrated threshold, recall per defect type. *(Insert numbers.)*

**Known limitations and failure modes.** *(Fill from `reports/.../failures/`.)*
Typical for this method: small low-contrast defects; misalignment or rotation
that the normal set did not cover; logical anomalies (missing or wrong parts)
that look locally normal; sensitivity to lighting changes. Scores are not
calibrated probabilities and are not comparable across categories.

**Ethical / licensing.** MVTec AD is CC BY-NC-SA 4.0: non-commercial use, with
attribution.
