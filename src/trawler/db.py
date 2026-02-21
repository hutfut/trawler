from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Generator

import psycopg
from psycopg.rows import dict_row

from trawler.config import get_config

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id              TEXT PRIMARY KEY,
    slug            TEXT,
    title           TEXT NOT NULL,
    description     TEXT,
    tags            JSONB DEFAULT '[]',
    start_date      TIMESTAMPTZ,
    end_date        TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS markets (
    id              TEXT PRIMARY KEY,
    event_id        TEXT REFERENCES events(id),
    question        TEXT NOT NULL,
    outcomes        JSONB NOT NULL,
    outcome_prices  JSONB,
    volume          DOUBLE PRECISION DEFAULT 0,
    volume_num      DOUBLE PRECISION DEFAULT 0,
    liquidity       DOUBLE PRECISION DEFAULT 0,
    closed_time     TIMESTAMPTZ,
    resolution      TEXT,
    asset_ids       JSONB DEFAULT '[]',
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS price_history (
    id              SERIAL PRIMARY KEY,
    market_id       TEXT NOT NULL REFERENCES markets(id),
    ts              TIMESTAMPTZ NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    UNIQUE (market_id, ts)
);
CREATE INDEX IF NOT EXISTS idx_price_history_market ON price_history(market_id);

CREATE TABLE IF NOT EXISTS scores (
    market_id       TEXT PRIMARY KEY REFERENCES markets(id),
    surprise        DOUBLE PRECISION,
    narrative_arc   DOUBLE PRECISION,
    absurdity       DOUBLE PRECISION,
    volume_score    DOUBLE PRECISION,
    significance    DOUBLE PRECISION,
    topical         DOUBLE PRECISION,
    composite       DOUBLE PRECISION NOT NULL,
    scored_at       TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS scripts (
    id              SERIAL PRIMARY KEY,
    market_ids      JSONB NOT NULL,
    format          TEXT NOT NULL DEFAULT 'compilation',
    script_text     TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now()
);
"""


@contextmanager
def get_conn() -> Generator[psycopg.Connection, None, None]:
    cfg = get_config()
    with psycopg.connect(cfg.database_url, row_factory=dict_row) as conn:
        yield conn


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(_SCHEMA_SQL)
        conn.commit()
