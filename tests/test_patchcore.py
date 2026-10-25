import numpy as np

from src.metrics import image_metrics, pixel_auroc
from src.models.patchcore import PatchCoreScorer, greedy_coreset, knn_distance


def _normal_maps(n, rng):
    return rng.normal(0, 1, (n, 8, 6, 6)).astype(np.float32)


def test_knn_distance_exact():
    bank = np.array([[0, 0], [3, 4]], np.float32)
    q = np.array([[0, 0], [3, 4], [6, 8]], np.float32)
    np.testing.assert_allclose(knn_distance(q, bank, 1), [0, 0, 5], atol=1e-4)


def test_coreset_size_and_uniqueness():
    x = np.random.default_rng(0).normal(size=(500, 16)).astype(np.float32)
    idx = greedy_coreset(x, 0.1, seed=0)
    assert len(idx) == 50 and len(set(idx.tolist())) == 50


def test_coreset_covers_outlier():
    x = np.zeros((200, 4), np.float32)
    x[123] = 100.0  # a far-away point must be selected by k-center
    assert 123 in greedy_coreset(x, 0.05, seed=1)


def test_scorer_detects_and_localises_injected_anomaly(tmp_path):
    rng = np.random.default_rng(0)
    scorer = PatchCoreScorer(coreset_ratio=0.5, seed=0).fit(_normal_maps(20, rng))
    normal_scores, _ = scorer.score(_normal_maps(10, rng), out_size=24)
    scorer.calibrate(normal_scores, 99)

    test = _normal_maps(4, rng)
    test[2:, :, 1, 4] += 8.0  # inject anomaly at patch (1, 4) in two images
    scores, maps = scorer.score(test, out_size=24, sigma=0)
    assert scores[2:].min() > scores[:2].max()
    assert (scores[2:] >= scorer.threshold).all()
    peak = np.unravel_index(maps[2].argmax(), maps[2].shape)
    assert 4 <= peak[0] <= 8 and 16 <= peak[1] <= 20  # patch (1,4) of 6x6 -> 4px cells

    scorer.save(tmp_path)
    loaded, meta = PatchCoreScorer.load(tmp_path)
    np.testing.assert_allclose(loaded.score(test, 24, 0)[0], scores)
    assert meta["threshold"] == scorer.threshold


def test_metrics():
    labels = np.array([0, 0, 1, 1])
    m = image_metrics(labels, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert m["image_auroc"] == 1.0 and m["f1"] == 1.0
    assert m["confusion"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    masks = np.zeros((1, 4, 4)); masks[0, 0, 0] = 1
    maps = np.zeros((1, 4, 4)); maps[0, 0, 0] = 1
    assert pixel_auroc(masks, maps) == 1.0
