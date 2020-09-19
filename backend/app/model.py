from datetime import datetime
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


class OHLCV(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    open = db.Column(db.Float)
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    close = db.Column(db.Float)
    volume = db.Column(db.Float)

    def __repr__(self):
        return f"{self.timestamp}: {self.open}, {self.high}, {self.low}, {self.close}, {self.volume}"


class ForecastHistory(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    forecast = db.Column(db.Float)
    
    def __repr__(self):
        return f"{self.timestamp}: {self.forecast}"
