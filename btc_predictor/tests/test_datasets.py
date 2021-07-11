from datetime import datetime

import pandas as pd

from btc_predictor.datasets.data_reader import BitfinexCandlesAPI


def test_bitfinex_candles_api():
    candles = BitfinexCandlesAPI()

    assert candles.period == '1m'
    assert candles.symbol == 'tBTCUSD'
    assert not candles.start_time
    assert not candles.end_time
    assert not candles.data

    candles.load(start_time=1610000000000, limit=10)

    assert candles.start_time == datetime.fromtimestamp(1610000000000 // 1000).strftime("%Y%m%d")
    assert isinstance(candles.data, pd.DataFrame)
    assert candles.data.shape[0] == 10
