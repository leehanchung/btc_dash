from typing import Any
import json
import pandas as pd
from btc_dash.websocket_manager import BitfinexSocketManager, WebsocketDataValidationError


bsm = BitfinexSocketManager()
ws = bsm.start_candle_socket()
ws_metadata = {}
columns = ['timestamp', 'open', 'close', 'high', 'low', 'volume']

class OrderedCircularBuffer:
    def __init__(self, buffer_size:int = 30):
        self.data = []#deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        # super(OrderedCircularBuffer, self).__init__(maxlen=size)

    def update(self, item: Any):
        """Updates the ordered circular buffer one item at a time.

        Args:
            item ([type]): [description]
        """
        if len(self.data) == 0:
            self.data.append(item)
        if len(self.data) < self.buffer_size:
            self.data.append(item)
        pass
    
    def __repr__(self):
        return str(self.data)

buffer = OrderedCircularBuffer()

while True:
    msg = ws.recv()
    msg = json.loads(msg)
    # print(f"{msg}")

    # first two message upon receiving are meta data logs:
    # {'event': 'info', 'version': 2, 'serverId': '4cb58267-fdae-4356-9937-3915befba341', 'platform': {'status': 1}}
    # {'event': 'subscribed', 'channel': 'candles', 'chanId': 219582, 'key': 'trade:1m:tBTCUSD'}
    if isinstance(msg, dict):
        for key, value in msg.items():
            if key != "event" and key in ws_metadata:
                raise WebsocketDataValidationError
                break
            else:
                ws_metadata[key] = value
        continue

    # other than headers, the results will be a list of length two, first item is chanId and
    # second component will be 
    #   1. string 'hb'
    #   2. array of length 1
    #   3. array of multiple length for the first msg
    # the timestamps will have duplicates as well
    if not isinstance(msg, list) or len(msg) != 2:
        raise WebsocketDataValidationError
        break

    print(f'[INFO] Stepping...')
    if not isinstance(msg[1], list):
        continue
    elif all(isinstance(item, list) for item in msg[1]):
        print("[DEBUG] Initial entry...")
        # new_entries = pd.DataFrame(msg[1], columns=columns)
        # print(msg[1])
        buffer.update(msg[1])
        # print(df_queue)
    elif all(isinstance(item, (int, float)) for item in msg[1]):
        print("[DEBUG] Regular entry...")
        buffer.update(msg[1])
        # new_entry = pd.DataFrame([msg[1]], columns=columns)
        # df_queue.append(new_entry)
    else:
        print(f'[ERROR] WTF {msg[1]}')
        raise WebsocketDataValidationError
        break
    
    print(buffer)

print('closing websocket...')
ws.close()
