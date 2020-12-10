import os
import pathlib
from pathlib import Path
from dotenv import load_dotenv


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent


class BaseConfig(object):
    """Base config"""
    DEBUG = False
    TESTING = False
    DB_SERVER = '0.0.0.0'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    @property
    def SQLALCHEMY_DATABASE_URI(self):         # Note: all caps
        return 'postgres://user@{}/foo'.format(self.DB_SERVER)


class TestingConfig(BaseConfig):
    DEBUG = True
    DB_SERVER = 'localhost'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    DB_SERVER = os.getenv("DB_SERVER")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URI")


class ProductionConfig(BaseConfig):
    TESTING = False
    DEBUG = False
    DB_SERVER = os.getenv("DB_SERVER")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URI")


def get_config() -> BaseConfig:
    """Getting config from sys env variable FLASK_ENV and choose the
    appropriate config object

    Returns:
        BaseConfig: the configuration corresponding to FLASK_ENV
    """
    env = os.environ.get("FLASK_ENV", 'development')

    if env == "production":
        env_path = Path('.')/'.env'
        load_dotenv(dotenv_path=env_path)
        config = ProductionConfig
    elif env == "testing":
        config = TestingConfig
    else:
        env_path = Path('.')/'.env.dev'
        load_dotenv(dotenv_path=env_path)
        config = DevelopmentConfig

    return config
