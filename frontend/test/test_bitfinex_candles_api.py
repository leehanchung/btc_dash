import pytest
from btc_dash.bitfinex_api import bitfinex_candles_api


def test_bitfinex_candles_api():
    data = bitfinex_candles_api()
    assert data.columns.to_list() == ["Open", "Close", "High", "Low", "Volume"]
    assert data.shape == (120, 5)
