from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row

_MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def _redact_secrets(text: str, dsn: str) -> str:
    """Remove the DSN and any password from a message before it is printed."""
    redacted = text.replace(dsn, "***")
    try:
        password = conninfo_to_dict(dsn).get("password")
    except Exception:
        password = None
    if password:
        redacted = redacted.replace(password, "***")
    return redacted


def _config_file_url() -> str:
    """Read database.url from config.yml without importing the app settings."""
    try:
        from agent.config import Settings
    except ImportError:
        return ""
    return Settings.load("config.yml").database.url.get_secret_value()


def _resolve_database_url(args_url: str | None) -> str:
    if args_url:
        return args_url
    env_url = os.getenv("DATABASE_URL", "").strip()
    if env_url:
        return env_url
    config_url = _config_file_url().strip()
    if config_url:
        return config_url
    raise SystemExit(
        "No database URL configured: pass --url or set DATABASE_URL"
    )


def _migration_sort_key(name: str) -> tuple[int, int, str]:
    """Order migrations by leading version numbers, not raw filename.

    ``0001_init.sql`` (1, 0) sorts before ``0001_1_provider_safe_embeddings.sql``
    (1, 1), which sorts before ``0002_...`` (2, 0). Plain filename sorting
    would put ``0001_1_...`` before ``0001_init...``.
    """
    base = name[:-4] if name.endswith(".sql") else name
    parts = base.split("_")

    def num(part: str) -> int:
        try:
            return int(part)
        except ValueError:
            return -1

    return (num(parts[0]), num(parts[1]) if len(parts) > 1 else 0, name)


def _applied_versions(conn: psycopg.Connection) -> set[str]:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT version FROM schema_migrations")
            return {row["version"] for row in cur.fetchall()}
    except psycopg.errors.UndefinedTable:
        conn.rollback()
        return set()


def run_migrations(dsn: str) -> int:
    applied = 0
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        known = _applied_versions(conn)
        for path in sorted(_MIGRATIONS_DIR.glob("*.sql"), key=lambda p: _migration_sort_key(p.name)):
            version = path.name
            if version in known:
                continue
            sql = path.read_text(encoding="utf-8")
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            conn.commit()
            print(f"Applied migration: {version}")
            applied += 1
    return applied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agent.db.migrate",
        description="Apply PostgreSQL schema migrations for the agent database.",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Database URL (overrides DATABASE_URL)",
    )
    args = parser.parse_args(argv)

    try:
        dsn = _resolve_database_url(args.url)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        applied = run_migrations(dsn)
    except Exception as exc:
        message = _redact_secrets(str(exc), dsn)
        print(f"Migration failed: {message}", file=sys.stderr)
        return 1

    print(f"Migrations applied: {applied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
