import logging

# import os

# import alembic.config
from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy_utils import database_exists, create_database

from app.config import Config  # , PACKAGE_ROOT

_logger = logging.getLogger("app.db")

Base = declarative_base()


def init_database(app: Flask, config: Config, db_session=None) -> None:
    """Connect to the database and attach DB session to the app."""

    if not db_session:
        engine = create_db_engine_from_config(config=config)
        db_session = create_db_session(engine=engine)

    app.db_session = db_session

    @app.teardown_appcontext
    def shutdown_session(exception=None):
        db_session.remove()


def create_db_engine_from_config(*, config: Config) -> Engine:
    """The Engine is the starting point for any SQLAlchemy application.

    It’s “home base” for the actual database and its DBAPI, delivered to the
    SQLAlchemy application through a connection pool and a Dialect, which
    describes how to talk to a specific kind of database / DBAPI combination.
    """

    db_url = config.SQLALCHEMY_DATABASE_URI
    print(f"[INFO] create_db_engine_from_config {db_url}")
    print(f"[INFO] {db_url} database exists? {database_exists(db_url)}")
    if not database_exists(db_url):
        print(f"[INFO] database doesnt exist at {db_url}, creating...")
        create_database(db_url)

    engine = create_engine(db_url)
    # _logger.info(f"creating DB conn with URI: {db_url}")
    return engine


def create_db_session(*, engine: Engine) -> scoped_session:
    """Broadly speaking, the Session establishes all conversations with the
    database.

    It represents a “holding zone” for all the objects which you’ve loaded or
    associated with it during its lifespan.
    """
    return scoped_session(
        sessionmaker(autocommit=False, autoflush=False, bind=engine)
    )


# def run_migrations():
#     """Run the DB migrations prior to the tests."""

#     # alembic looks for the migrations in the current
#     # directory so we change to the correct directory.
#     os.chdir(str(ROOT))
#     alembicArgs = ["--raiseerr", "upgrade", "head"]
#     alembic.config.main(argv=alembicArgs)
