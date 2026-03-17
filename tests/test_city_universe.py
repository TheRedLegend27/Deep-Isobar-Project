from deep_isobar.data.city_universe import get_city_profile


def test_get_city_profile():
    profile = get_city_profile("Chicago")
    assert profile.city_code == "CHI"