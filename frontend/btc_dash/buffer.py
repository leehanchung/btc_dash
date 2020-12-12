from typing import List, Union


class OrderedCircularBuffer:
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
                pass
        else:
            if self.data[-1][0] < item[0]:
                self.data.pop(0)
                self.data.append(item)

    def __repr__(self):
        return str(self.data)
