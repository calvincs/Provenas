"""store — the persistent spine of the fabric (SQLite).

The durable source of truth: triples, rules, and an append-only audit log. Everything the engines
need is loaded into the in-memory KnowledgeGraph on demand (to_kg) so the existing exact engines
(infer / kgvm / solver) run unchanged; SQLite gives persistence, indexing, transactions, and a
ledger of every change. Single-writer (perfect for one box; not high-concurrency).
"""
from __future__ import annotations

import json
import sqlite3
import time

from provenas.infer import Rule
from provenas.kg import KnowledgeGraph


def _to_tuple(x):
    """JSON round-trips tuples as lists; rewrite terms must be tuples (for ==/matching)."""
    return tuple(_to_tuple(e) for e in x) if isinstance(x, list) else x


class Store:
    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(path)
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS triples(s TEXT,r TEXT,o TEXT,source TEXT,ts REAL,"
            " UNIQUE(s,r,o));"
            "CREATE TABLE IF NOT EXISTS rules(name TEXT PRIMARY KEY,body TEXT,head TEXT,source TEXT,ts REAL);"
            "CREATE TABLE IF NOT EXISTS tools(name TEXT PRIMARY KEY,src TEXT,tests TEXT,source TEXT,ts REAL);"
            "CREATE TABLE IF NOT EXISTS rewrites(name TEXT PRIMARY KEY,rule TEXT,source TEXT,ts REAL);"
            "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);"
            "CREATE TABLE IF NOT EXISTS log(ts REAL,kind TEXT,detail TEXT);"
        )
        self.db.commit()

    # ---- facts ----
    def assert_(self, s, r, o, source="user"):
        self.db.execute("INSERT OR IGNORE INTO triples VALUES(?,?,?,?,?)", (s, r, o, source, time.time()))
        self.db.commit()
        return self

    def retract(self, s, r, o):
        self.db.execute("DELETE FROM triples WHERE s=? AND r=? AND o=?", (s, r, o))
        self.db.commit()
        return self

    def triples(self):
        return [(s, r, o) for s, r, o in self.db.execute("SELECT s,r,o FROM triples")]

    # ---- rules ----
    def add_rule(self, rule, source="user"):
        self.db.execute("INSERT OR REPLACE INTO rules VALUES(?,?,?,?,?)",
                        (rule.name, json.dumps(rule.body), json.dumps(rule.head), source, time.time()))
        self.db.commit()
        return self

    def rules(self):
        out = []
        for name, body, head, *_ in self.db.execute("SELECT * FROM rules"):
            patterns = [tuple(p) for p in json.loads(body)]
            out.append(Rule(patterns, tuple(json.loads(head)), name))
        return out

    # ---- tools (synthesized Python functions) ----
    def add_tool(self, name, src, tests, source="qwen"):
        self.db.execute("INSERT OR REPLACE INTO tools VALUES(?,?,?,?,?)",
                        (name, src, json.dumps(tests), source, time.time()))
        self.db.commit()
        return self

    def tools(self):
        return [(name, src, json.loads(tests))
                for name, src, tests, *_ in self.db.execute("SELECT * FROM tools")]

    def get_tool(self, name):
        row = self.db.execute("SELECT src FROM tools WHERE name=?", (name,)).fetchone()
        return row[0] if row else None

    # ---- rewrite rules (term-rewriting rules, shared in the same KB) ----
    def add_rewrite(self, name, rule, source="user"):
        self.db.execute("INSERT OR REPLACE INTO rewrites VALUES(?,?,?,?)",
                        (name, json.dumps(rule), source, time.time()))
        self.db.commit()
        return self

    def rewrites(self):
        return [_to_tuple(json.loads(r)) for _, r in self.db.execute("SELECT name,rule FROM rewrites")]

    # ---- materialize / log / status ----
    def to_kg(self):
        kg = KnowledgeGraph()
        for s, r, o in self.triples():
            kg.assert_(s, r, o)
        return kg

    def log(self, kind, detail):
        self.db.execute("INSERT INTO log VALUES(?,?,?)",
                        (time.time(), kind, detail if isinstance(detail, str) else json.dumps(detail)))
        self.db.commit()

    def recent_log(self, n=20):
        return list(self.db.execute("SELECT ts,kind,detail FROM log ORDER BY ts DESC LIMIT ?", (n,)))

    def set_meta(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, value))
        self.db.commit()

    def get_meta(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def relations(self):
        rs = {r for _, r, _ in self.triples()}
        for rule in self.rules():
            rs.add(rule.head[1])
            for atom in rule.body:
                if atom[1] != "!=":
                    rs.add(atom[1])
        return rs

    def entities(self):
        es = set()
        for s, _, o in self.triples():
            es.add(s)
            es.add(o)
        return es

    def counts(self):
        t = self.db.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        r = self.db.execute("SELECT COUNT(*) FROM rules").fetchone()[0]
        return t, r

    def close(self):
        self.db.close()
