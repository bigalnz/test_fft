import os

DB_CONN_STRING = os.environ.get("KIWITRACKER_DB", "sqlite:///./main.db")
