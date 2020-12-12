import pytest
from btc_dash.buffer import OrderedCircularBuffer


def test_inorder_update():
    buffer = OrderedCircularBuffer()
    assert not buffer.data

    data = [[n for n in range(m, m+5)] for m in range(35)]
    for d in data:
        buffer.update(d)
    assert len(buffer.data) == 30
    assert buffer.data[0] == [5, 6, 7, 8, 9]
    assert buffer.data[-1] == [34, 35, 36, 37, 38]
    assert buffer.data == [[n for n in range(m, m+5)] for m in range(5, 35)]
    assert buffer.data == sorted(buffer.data, key=lambda x: x[0])


def test_duplicateorder_update():
    buffer = OrderedCircularBuffer()
    data = [[n for n in range(m, m+5)] for m in range(35)]
    for idx, d in enumerate(data):
        if idx % 2:
            buffer.update(d)
        buffer.update(d)

    assert len(buffer.data) == 30
    assert buffer.data[0] == [5, 6, 7, 8, 9]
    assert buffer.data[-1] == [34, 35, 36, 37, 38]
    assert buffer.data == [[n for n in range(m, m+5)] for m in range(5, 35)]
    assert buffer.data == sorted(buffer.data, key=lambda x: x[0])


def test_outoforder_update():
    buffer = OrderedCircularBuffer()       
    data = [[n for n in range(m, m+5)] for m in range(35)]
    for idx, d in enumerate(data):
        buffer.update(d)
    assert len(buffer.data) == 30
    assert buffer.data == [[n for n in range(m, m+5)] for m in range(5, 35)]
    assert buffer.data == sorted(buffer.data, key=lambda x: x[0])

    buffer.update([33, 34, 35, 36, 37])
    assert buffer.data == sorted(buffer.data, key=lambda x: x[0])
    buffer.update([34, 35, 36, 37, 38])
    assert buffer.data == sorted(buffer.data, key=lambda x: x[0])
    buffer.update([35, 36, 37, 38, 39])
    buffer.update([34, 35, 36, 37, 38])
    buffer.update([34, 35, 36, 37, 38])
    buffer.update([35, 36, 37, 38, 39])
    buffer.update([35, 36, 37, 38, 39])
    buffer.update([34, 35, 36, 37, 38])
    assert buffer.data == sorted(buffer.data, key=lambda x: x[0])
    assert buffer.data[-1] == [35, 36, 37, 38, 39]
    buffer.update([35, 36, 37, 38, 39])
    buffer.update([34, 35, 36, 37, 38])
    buffer.update([36, 37, 38, 39, 40])
    buffer.update([35, 36, 37, 38, 39])
    buffer.update([34, 35, 36, 37, 38])
    buffer.update([36, 37, 38, 39, 40])
    buffer.update([36, 37, 38, 39, 40])
    buffer.update([36, 37, 38, 39, 40])
    buffer.update([36, 37, 38, 39, 40])
    assert len(buffer.data) == 30
    assert buffer.data[-1] == [36, 37, 38, 39, 40]
    assert buffer.data[0] == [7, 8, 9, 10, 11]
    assert buffer.data == sorted(buffer.data, key=lambda x: x[0])