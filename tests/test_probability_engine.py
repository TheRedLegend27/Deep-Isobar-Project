from deep_isobar.models.probability_engine import probability_ge_normal


def test_probability_ge_normal_bounds():
    p = probability_ge_normal(84.0, 2.0, 85.0)
    assert 0.0 <= p <= 1.0