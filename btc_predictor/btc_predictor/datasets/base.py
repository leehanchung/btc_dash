import logging
from abc import ABC, abstractmethod

import pandas as pd

_logger = logging.getLogger(__name__)


class DataReadingError(Exception):
    pass


class BaseDataset(ABC):
    """Base Dataset template for datasets. We characterize btcusd time series
    data using source, start_time, end_time, and resolution.
    """

    start_time: str
    end_time: str
    resolution: str
    data: pd.DataFrame

    @abstractmethod
    def load(
        self,
        *,
        filename: str = None,
        start_time: int = None,
        limit: int = 10000
    ) -> None:
        pass

    @property
    def pd(self) -> pd.DataFrame:
        return self.data
