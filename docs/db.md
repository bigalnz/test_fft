## Database

Kiwitracker uses [`SQLAlchemy`](https://www.sqlalchemy.org/) with `SQLite` backend for database handling.
For migrations [`alembic`](https://alembic.sqlalchemy.org/en/latest/) is used.

Default db name is `main.db`.

## Useful commands:

Creating new revision:

```
alembic revision --autogenerate -m "<message>"
```

Upgrade to latest DB revision:

```
alembic upgrade head
```
