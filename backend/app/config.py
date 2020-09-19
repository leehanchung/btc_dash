import os
import pathlib
from dotenv import load_dotenv


load_dotenv()
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
    TESTING = False
    DB_SERVER = 'localhost'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    TESTING = False
    DB_SERVER = ""


class ProductionConfig(BaseConfig):
    DB_SERVER = ""


def get_config() -> BaseConfig:
    env = os.environ.get("FLASK_ENV", 'testing')
    if  env == "production":
        config = ProductionConfig
    elif env == "development":
        config = DevelopmentConfig
    else:
        os.environ["FLASK_ENV"] = "testing"
        config = TestingConfig

    print(f"[INFO] Running using {env} config...")
    return config