from sqlalchemy import Column, DateTime, Float
from sqlalchemy.sql import func

from app.db.core import Base


class Forecast(Base):
    __tablename__ = "forecast_observed"
    timestamp = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        primary_key=True,
        index=True,
    )
    forecast = Column(Float)
    observed = Column(Float)

    def __repr__(self):
        return f"{self.timestamp}: {self.forecast}, {self.observed}"
