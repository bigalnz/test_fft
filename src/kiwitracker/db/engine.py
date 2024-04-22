import os
from pathlib import Path

from alembic.command import upgrade
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.engine import Engine, create_engine

_ENGINE: Engine | None = None


def construct_db_connection_string(db_file=None):
    _env_str = os.environ.get("KIWITRACKER_DB")

    if not _env_str:
        p = os.path.abspath("./main.db" if db_file is None else db_file)
        _env_str = f"sqlite:///{p}"

    return _env_str


def construct_sqlalchemy_engine(db_file=None):
    global _ENGINE

    if not _ENGINE:
        url = construct_db_connection_string(db_file)
        _ENGINE = create_engine(url)


def get_sqlalchemy_engine() -> Engine:
    return _ENGINE


def _get_alembic_dir() -> str:
    return Path(__file__).parent / "migrations"


def _get_alembic_config(url: str) -> Config:
    alembic_dir = _get_alembic_dir()
    alembic_ini_path = alembic_dir / "alembic.ini"
    alembic_cfg = Config(alembic_ini_path)
    alembic_cfg.set_main_option("script_location", str(alembic_dir))
    url = url.replace("%", "%%")
    alembic_cfg.set_main_option("sqlalchemy.url", url)
    return alembic_cfg


def migrate_if_needed(engine: Engine, revision: str) -> None:
    alembic_cfg = _get_alembic_config(engine.url.render_as_string(hide_password=False))
    script_dir = ScriptDirectory.from_config(alembic_cfg)
    with engine.begin() as conn:
        context = MigrationContext.configure(conn)
        if context.get_current_revision() != script_dir.get_current_head():
            upgrade(alembic_cfg, revision)
