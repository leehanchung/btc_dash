import os

# import logging
import pathlib
from pathlib import Path
from dotenv import load_dotenv


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent


class Config(object):
    """Base config"""

    DEBUG = False
    TESTING = False

    DB_USER = os.getenv("DB_USER", "user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "pass")
    DB_HOST = os.getenv("DB_HOST", "localhost:5432")
    DB_NAME = os.getenv("DB_NAME", "dev_db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    @property
    def SQLALCHEMY_DATABASE_URI(self):  # Note: all caps
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}/{self.DB_NAME}"
        )


class TestingConfig(Config):
    DEBUG = True

    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


class DevelopmentConfig(Config):
    DEBUG = True

    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_NAME = os.getenv("DB_NAME")


class ProductionConfig(Config):
    TESTING = False
    DEBUG = False

    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_NAME = os.getenv("DB_NAME")


def get_config() -> Config:
    """Getting config from sys env variable FLASK_ENV and choose the
    appropriate config object

    Returns:
        BaseConfig: the configuration corresponding to FLASK_ENV
    """
    env = os.environ.get("FLASK_ENV", "development")

    if env == "production":
        env_path = Path(".") / ".env"
        load_dotenv(dotenv_path=env_path)
        config = ProductionConfig()
    elif env == "testing":
        config = TestingConfig()
    else:
        env_path = Path(".") / ".env.dev"
        load_dotenv(dotenv_path=env_path)
        config = DevelopmentConfig()

    return config
