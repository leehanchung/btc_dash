from sqlalchemy import Column, DateTime, Float
from sqlalchemy.sql import func

from app.db.core import Base


class OHLCV(Base):
    __tablename__ = "historic_ohlcv_data"
    timestamp = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        primary_key=True,
        index=True,
    )
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

    def __repr__(self):
        return (f"{self.timestamp}: {self.open}, {self.high}, {self.low},"
                f" {self.close}, {self.volume}")
