import json
from app.websockets import BitfinexSocketManager


bsm = BitfinexSocketManager()
ws = bsm.start_candle_socket()

while True:
    result = ws.recv()
    result = json.loads(result)

    print(f"{result}")

ws.close()