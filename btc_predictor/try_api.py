import json
import logging
import time

import requests
from api import app

_logger = logging.getLogger(__name__)


def test_local_api():
    URL = "http://127.0.0.1:3000/invoke"
    body = [
        35581,
        35787,
        35702,
        35786,
        35576,
        35427,
        35356.49956215,
        35499,
        35417,
        35461,
        35500,
        35535,
        35620,
        35466,
        35422,
        35385.847202,
        35389,
    ]

    response = requests.post(URL, json=json.loads(f"{body}"))
    _logger.info(response.status_code)
    _logger.info(response.text)


def test_local_app(*, loop: int = 100):
    body = [
        35581,
        35787,
        35702,
        35786,
        35576,
        35427,
        35356.49956215,
        35499,
        35417,
        35461,
        35500,
        35535,
        35620,
        35466,
        35422,
        35385.847202,
        35389,
    ]

    context = {}
    event = {"body": json.dumps(body)}

    start = time.time()
    for _ in range(loop):
        response = app.lambda_handler(event, context)
        _logger.info(f"{response}")
    end = time.time()
    _logger.info(f"avg inference time: {(end-start)*1000/loop:.4f}ms")


def main():
    # test_local_app()
    test_local_api()


if __name__ == "__main__":
    main()
