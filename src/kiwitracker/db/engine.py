from sqlalchemy.engine import Engine, create_engine

from kiwitracker.config import DB_CONN_STRING

_ENGINE: Engine | None = None


def get_sqlalchemy_engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(DB_CONN_STRING)
    return _ENGINE
