import json

from btc_dash.bitfinex_api import (
    BitfinexSocketManager,
    BitfinexWebsocketError,
    OrderedBuffer,
)


def main():
    bsm = BitfinexSocketManager()
    ws = bsm.start_candle_socket()
    ws_metadata = {}
    # columns = ['timestamp', 'open', 'close', 'high', 'low', 'volume']
    buffer = OrderedBuffer()

    while True:
        msg = ws.recv()
        msg = json.loads(msg)
        # first two message upon receiving are meta data logs:
        # {'event': 'info',
        #  'version': 2,
        #  'serverId': '4cb58267-fdae-4356-9937-3915befba341',
        #  'platform': {'status': 1}
        # }
        # {'event': 'subscribed',
        #  'channel': 'candles',
        #  'chanId': 219582,
        # 'key': 'trade:1m:tBTCUSD'
        # }
        if isinstance(msg, dict):
            for key, value in msg.items():
                if key != "event" and key in ws_metadata:
                    raise BitfinexWebsocketError
                    break
                else:
                    ws_metadata[key] = value
            continue

        # Other than headers, the results will be a list of length two, first
        # itemis chanId andsecond component will be
        #   1. string 'hb'
        #   2. array of length 1
        #   3. array of multiple length for the first msg
        # the timestamps will have duplicates as well
        if not isinstance(msg, list) or len(msg) != 2:
            raise BitfinexWebsocketError
            break

        print("[INFO] Stepping...")
        if not isinstance(msg[1], list):
            continue
        elif all(isinstance(item, list) for item in msg[1]):
            print("[DEBUG] Initial entries...")
            for item in sorted(msg[1], key=lambda x: x[0]):
                buffer.update(item)
        elif all(isinstance(item, (int, float)) for item in msg[1]):
            print("[DEBUG] Regular entry...")
            buffer.update(msg[1])
        else:
            print(f"[ERROR] WTF {msg[1]}")
            raise BitfinexWebsocketError
            break

        print(buffer)

    print("closing websocket...")
    ws.close()


if __name__ == "__main__":
    main()
