from datetime import date

from ai_trader.data.binance import _iter_months, _url


def test_iter_months_spans_year_boundary():
    got = list(_iter_months(date(2023, 11, 1), date(2024, 2, 15)))
    assert got == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]


def test_iter_months_single():
    assert list(_iter_months(date(2024, 6, 1), date(2024, 6, 30))) == [(2024, 6)]


def test_iter_months_ordered_endpoints_required():
    assert list(_iter_months(date(2024, 6, 1), date(2024, 5, 1))) == []


def test_url_format():
    url = _url("ETHUSDT", "1m", 2024, 3)
    assert url.endswith("/ETHUSDT/1m/ETHUSDT-1m-2024-03.zip")
    assert url.startswith("https://data.binance.vision/data/spot/monthly/klines/")
