import math

import pytest

from tinyllm.schedule import lr_at

BASE, MIN, WARMUP, TOTAL = 6e-4, 6e-5, 200, 61_000


def test_first_step_is_one_warmup_fraction():
    assert lr_at(0, TOTAL, BASE, MIN, WARMUP) == pytest.approx(BASE / WARMUP)


def test_warmup_ends_at_base_lr():
    assert lr_at(WARMUP - 1, TOTAL, BASE, MIN, WARMUP) == pytest.approx(BASE)


def test_cosine_starts_at_base_lr():
    # warmup 직후 progress=0 이라 cos(0)=1 -> min + (base-min) = base.
    assert lr_at(WARMUP, TOTAL, BASE, MIN, WARMUP) == pytest.approx(BASE)


def test_ends_at_min_lr():
    assert lr_at(TOTAL, TOTAL, BASE, MIN, WARMUP) == pytest.approx(MIN)


def test_clamps_past_the_end():
    assert lr_at(TOTAL * 2, TOTAL, BASE, MIN, WARMUP) == pytest.approx(MIN)


def test_monotone_decreasing_after_warmup():
    steps = range(WARMUP, TOTAL, 1000)
    values = [lr_at(s, TOTAL, BASE, MIN, WARMUP) for s in steps]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_midpoint_is_halfway_in_cosine_terms():
    mid = lr_at(WARMUP + (TOTAL - WARMUP) // 2, TOTAL, BASE, MIN, WARMUP)
    assert mid == pytest.approx(MIN + 0.5 * (BASE - MIN), rel=1e-3)
