import json
import threading
from websocket import create_connection, WebSocket


class BitfinexSocketManager(threading.Thread):

    STREAM_URL = "wss://api-pub.bitfinex.com/ws/2"

    def __init__(self):
        threading.Thread.__init__(self)

    def start_candle_socket(self, timeframe:str = '1m') -> WebSocket:
        """Opens bitfinex public websocket channels for candles and subscribe
        to the BTCUSD pair. To listen to the websocket, use ws.recv() and
        convert the response to JSON.

        Data Format:
        CHANNEL_ID	int	Identification number assigned to the channel for the
                        duration of this connection.
        MTS	int	millisecond time stamp
        OPEN	float	First execution during the time frame
        CLOSE	float	Last execution during the time frame
        HIGH	float	Highest execution during the time frame
        LOW	float	Lowest execution during the timeframe
        VOLUME	float	Quantity of symbol traded within the timeframe

        https://docs.bitfinex.com/reference#ws-public-candles

        Args:
            timeframe (str, optional): Defaults to '1m'.

        Returns:
            WebSocket: [description]
        """
        ws = create_connection(self.STREAM_URL)
        print(type(ws))
        subscription = json.dumps({
            "event": "subscribe",
            "channel": "candles",
            "key": f"trade:{timeframe}:tBTCUSD"
        })
        ws.send(subscription)

        return ws

    def close_socket(self):
        raise NotImplementedError