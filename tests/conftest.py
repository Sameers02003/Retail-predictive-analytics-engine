"""Shared test setup.

The API and feature tests run against a small temporary COPY of the database
(only the tables the API needs), so tests never change your real data and use
little disk space. Each copy is deleted after its test.
"""

import gc
import importlib
import shutil
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_DB = PROJECT_ROOT / "data" / "db" / "retail.db"
TABLES_FOR_TESTS = ["clean_stores", "clean_orders", "clean_order_items", "forecast_evaluation", "anomaly_flags"]


@pytest.fixture(scope="session")
def template_db(tmp_path_factory):
    """Build one slim copy of the database, once per test run."""
    if not REAL_DB.exists():
        pytest.skip("Database not built yet: run the pipeline first")
    template = tmp_path_factory.mktemp("template") / "template.db"
    connection = sqlite3.connect(template)
    connection.execute("ATTACH DATABASE ? AS real", (str(REAL_DB),))
    for table in TABLES_FOR_TESTS:
        connection.execute(f"CREATE TABLE {table} AS SELECT * FROM real.{table}")
    connection.commit()
    connection.execute("DETACH DATABASE real")
    connection.close()
    return template


@pytest.fixture()
def test_db(template_db, tmp_path, monkeypatch):
    """A fresh copy of the slim database for one test."""
    db_copy = tmp_path / "retail_test.db"
    shutil.copy(template_db, db_copy)
    monkeypatch.setenv("DB_PATH", str(db_copy))
    yield db_copy
    gc.collect()                      # close any database connections still open
    try:
        db_copy.unlink(missing_ok=True)
    except OSError:
        pass                          # Windows may still hold the file; pytest removes it later


@pytest.fixture()
def api(test_db):
    import src.api
    return importlib.reload(src.api)


@pytest.fixture()
def client(api):
    return TestClient(api.app)


@pytest.fixture()
def conn(test_db):
    connection = sqlite3.connect(test_db)
    yield connection
    connection.close()