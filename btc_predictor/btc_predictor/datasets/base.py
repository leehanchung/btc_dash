import logging
from abc import ABC, abstractmethod

import pandas as pd

_logger = logging.getLogger(__name__)


class DataReadingError(Exception):
    pass


class Dataset(ABC):
    @abstractmethod
    def load(self, *, csv_file: str) -> None:
        pass

    @property
    def pd(self) -> pd.DataFrame:
        return getattr(self, "data", None)

    @property
    def start_time(self) -> str:
        return getattr(self, "start_time", None)

    @property
    def end_time(self) -> str:
        return getattr(self, "end_time", None)

    @property
    def resolution(self) -> str:
        return getattr(self, "resolution", None)
