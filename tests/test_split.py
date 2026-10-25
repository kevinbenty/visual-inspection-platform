from pathlib import Path

from src.data import Sample, split_train


def test_split_is_disjoint_and_reproducible():
    train = [Sample(Path(f"{i}.png"), 0, "good", None) for i in range(100)]
    fit1, cal1 = split_train(train, 0.1, seed=3)
    fit2, cal2 = split_train(train, 0.1, seed=3)
    assert fit1 == fit2 and cal1 == cal2
    assert len(cal1) == 10 and len(fit1) == 90
    assert not {s.path for s in fit1} & {s.path for s in cal1}
