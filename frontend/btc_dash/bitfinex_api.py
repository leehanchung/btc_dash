import json
import logging
import threading
from typing import List, Union

import pandas as pd
import requests

from websocket import create_connection, WebSocket


_logger = logging.getLogger(__name__)


class BitfinexSocketManager(threading.Thread):

    STREAM_URL = "wss://api-pub.bitfinex.com/ws/2"

    def __init__(self):
        threading.Thread.__init__(self)

    def start_candle_socket(self, timeframe: str = "1m") -> WebSocket:
        """Opens bitfinex public websocket channels for candles and subscribe
        to the BTCUSD pair. To listen to the websocket, use ws.recv() and
        convert the response to JSON.

        Data Format:
        CHANNEL_ID	int	    Identification number assigned to the channel for
                            the duration of this connection.
        MTS         int	    millisecond time stamp
        OPEN	    float	First execution during the time frame
        CLOSE   	float	Last execution during the time frame
        HIGH	    float	Highest execution during the time frame
        LOW 	    float	Lowest execution during the timeframe
        VOLUME  	float	Quantity of symbol traded within the timeframe

        https://docs.bitfinex.com/reference#ws-public-candles

        Args:
            timeframe (str, optional): Defaults to '1m'.

        Returns:
            WebSocket: [description]
        """
        ws = create_connection(self.STREAM_URL)

        subscription = json.dumps(
            {
                "event": "subscribe",
                "channel": "candles",
                "key": f"trade:{timeframe}:tBTCUSD",
            }
        )
        ws.send(subscription)
        return ws

    def close_socket(self):
        raise NotImplementedError


class OrderedBuffer:
    def __init__(self, buffer_size: int = 30):
        self.data = []
        self.buffer_size = buffer_size

    def update(self, item: List[Union[int, float]]):
        """Updates the ordered circular buffer one item at a time. First
        element of the item is expected to be UNIX Timestamp that will be used
        for indexing.

        Args:
            item ([List[Union[int, float]]]): timestamp + ohlcv
        """
        if len(self.data) < self.buffer_size:
            if not self.data:
                self.data.append(item)
            elif self.data[-1][0] < item[0]:
                self.data.append(item)
        else:
            if self.data[-1][0] < item[0]:
                self.data.pop(0)
                self.data.append(item)

    def __repr__(self):
        return str(self.data)


class BitfinexWebsocketError(Exception):
    pass


class BitfinexAPIError(Exception):
    pass


def bitfinex_candles_api(
    *, time_frame: str = "1m", symbol: str = "tBTCUSD", section: str = "hist"
) -> pd.DataFrame:
    """Get candles from bitfinex candles API. Expected periods is 120 and
    expected data format is list of list of int/float.

    https://docs.bitfinex.com/reference#rest-public-candles

    Args:
        time_frame (str, optional): [description]. Defaults to '1m'.
        symbol (str, optional): [description]. Defaults to "tBTCUSD".
        section (str, optional): [description]. Defaults to "hist".

    Raises:
        BitfinexAPIError

    Returns:
        pd.DataFrame: [description]
    """
    base_url = "https://api-pub.bitfinex.com/v2/candles/"
    url = base_url + f"trade:{time_frame}:{symbol}/{section}"
    columns = ["Timestamp", "Open", "Close", "High", "Low", "Volume"]

    response = requests.get(url)
    if not response.ok:
        raise BitfinexAPIError

    data = json.loads(response.text)
    assert all(isinstance(item, list) for item in data)
    data = sorted(data, key=lambda x: x[0])
    df = pd.DataFrame(data, columns=columns)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df = df.set_index("Timestamp")
    # TODO: tried to set Timestamp index frequency, but somehow getting
    # another row of data. Breaking all the Dash callbacks. Fuck.
    #
    # df = df.reset_index(drop=True)
    # df = df.set_index("Timestamp").asfreq("T")
    # print(df.shape)
    del response, data

    return df
