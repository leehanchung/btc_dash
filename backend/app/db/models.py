from datetime import datetime

from sqlalchemy import Column, DateTime, Float

# from sqlalchemy.dialects.postgresql import JSONB
# from sqlalchemy.sql import func

from api.persistence.core import Base

# db = SQLAlchemy()


class OHLCV(Base):
    timestamp = Column(
        DateTime, index=True, default=datetime.utcnow, primary_key=True
    )
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

    def __repr__(self):
        return (
            f"{self.timestamp}: {self.open}, {self.high}, {self.low},",
            f" {self.close}, {self.volume}",
        )


class ForecastHistory(Base):
    timestamp = Column(
        DateTime, index=True, default=datetime.utcnow, primary_key=True
    )
    forecast = Column(Float)

    def __repr__(self):
        return f"{self.timestamp}: {self.forecast}"
