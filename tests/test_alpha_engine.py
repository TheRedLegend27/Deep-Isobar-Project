from src.trading.alpha_engine import compute_alpha


def test_alpha():

    alpha = compute_alpha(0.6, 0.5)

    assert alpha == 0.1

from deep_isobar.trading.alpha_engine import compute_alpha


def test_compute_alpha():
    assert compute_alpha(0.7, 0.5) == 0.2