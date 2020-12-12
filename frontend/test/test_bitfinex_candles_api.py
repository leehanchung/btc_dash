import pytest
from btc_dash.bitfinex_api import bitfinex_candles_api


def test_bitfinex_candles_api():
    data = bitfinex_candles_api()
    assert len(data) == 120
    assert data.shape == (120, 6)