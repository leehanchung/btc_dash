from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import get_config


config = get_config()
engine = create_engine(config.SQLALCHEMY_DATABASE_URI, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
