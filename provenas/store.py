"""store — the persistent spine of the fabric (SQLite).

The durable source of truth: triples, rules, tools, rewrites, regression cases, an optional
relation schema, and an append-only audit log. The derived closure (all facts the rules imply,
with provenance) is MATERIALIZED: it is computed when facts or rules change and served from
cache on reads, so queries cost milliseconds regardless of rule depth.

Single-writer by design (one box, one process); a per-store lock + SQLite WAL make it safe to
share across threads (the HTTP service mode).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time

from provenas.infer import GUARDS, Rule, forward_chain_prov
from provenas.kg import KnowledgeGraph


def _to_tuple(x):
    """JSON round-trips tuples as lists; rewrite terms must be tuples (for ==/matching)."""
    return tuple(_to_tuple(e) for e in x) if isinstance(x, list) else x


class Store:
    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.RLock()
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS triples(s TEXT,r TEXT,o TEXT,source TEXT,ts REAL,"
            " UNIQUE(s,r,o));"
            "CREATE TABLE IF NOT EXISTS rules(name TEXT PRIMARY KEY,body TEXT,head TEXT,source TEXT,ts REAL);"
            "CREATE TABLE IF NOT EXISTS tools(name TEXT PRIMARY KEY,src TEXT,tests TEXT,source TEXT,ts REAL);"
            "CREATE TABLE IF NOT EXISTS rewrites(name TEXT PRIMARY KEY,rule TEXT,source TEXT,ts REAL);"
            "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);"
            "CREATE TABLE IF NOT EXISTS log(ts REAL,kind TEXT,detail TEXT);"
            "CREATE TABLE IF NOT EXISTS derived(s TEXT,r TEXT,o TEXT,rule TEXT,premises TEXT,"
            " UNIQUE(s,r,o));"
            "CREATE TABLE IF NOT EXISTS cases(name TEXT PRIMARY KEY,action TEXT,expect TEXT,ts REAL);"
            "CREATE TABLE IF NOT EXISTS schema_rels(rel TEXT PRIMARY KEY,doc TEXT);"
        )
        if "active" not in [c[1] for c in self.db.execute("PRAGMA table_info(rules)")]:
            self.db.execute("ALTER TABLE rules ADD COLUMN active INTEGER DEFAULT 1")  # migrate old KBs
        self.db.commit()
        self._cache = None                              # (kb_rev, kg, prov) — in-memory closure

    def _touch(self):
        """Facts or rules changed: bump the KB revision (invalidates the materialized closure)."""
        rev = int(self.get_meta("kb_rev", "0")) + 1
        self.db.execute("INSERT OR REPLACE INTO meta VALUES('kb_rev',?)", (str(rev),))
        self._cache = None

    # ---- facts ----
    def assert_(self, s, r, o, source="user"):
        with self.lock:
            self._check_schema(r)
            cur = self.db.execute("INSERT OR IGNORE INTO triples VALUES(?,?,?,?,?)",
                                  (s, r, o, source, time.time()))
            if cur.rowcount:
                self._touch()
            self.db.commit()
        return self

    def retract(self, s, r, o):
        with self.lock:
            cur = self.db.execute("DELETE FROM triples WHERE s=? AND r=? AND o=?", (s, r, o))
            if cur.rowcount:
                self._touch()
            self.db.commit()
        return self

    def triples(self):
        return [(s, r, o) for s, r, o in self.db.execute("SELECT s,r,o FROM triples")]

    # ---- rules (versioned by activation: disabled rules stay on record, stop firing) ----
    def add_rule(self, rule, source="user"):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO rules(name,body,head,source,ts,active)"
                            " VALUES(?,?,?,?,?,1)",
                            (rule.name, json.dumps(rule.body), json.dumps(rule.head), source, time.time()))
            self._touch()
            self.db.commit()
        return self

    def rules(self, all=False):
        q = "SELECT name,body,head FROM rules" + ("" if all else " WHERE active=1")
        return [Rule([tuple(p) for p in json.loads(body)], tuple(json.loads(head)), name)
                for name, body, head in self.db.execute(q)]

    def set_rule_active(self, name, active):
        with self.lock:
            cur = self.db.execute("UPDATE rules SET active=? WHERE name=?", (1 if active else 0, name))
            if cur.rowcount:
                self._touch()
                self.log("enable_rule" if active else "disable_rule", name)
            self.db.commit()
            return bool(cur.rowcount)

    # ---- tools (synthesized Python functions) ----
    def add_tool(self, name, src, tests, source="qwen"):
        with self.lock:
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
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO rewrites VALUES(?,?,?,?)",
                            (name, json.dumps(rule), source, time.time()))
            self.db.commit()
        return self

    def rewrites(self):
        return [_to_tuple(json.loads(r)) for _, r in self.db.execute("SELECT name,rule FROM rewrites")]

    # ---- regression cases (pinned question/answer pairs every rule change must preserve) ----
    def add_case(self, name, action, expect):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO cases VALUES(?,?,?,?)",
                            (name, json.dumps(action), json.dumps(expect), time.time()))
            self.db.commit()
        return self

    def cases(self):
        return [(n, json.loads(a), json.loads(e))
                for n, a, e in self.db.execute("SELECT name,action,expect FROM cases")]

    # ---- relation schema (declared relations; strict mode rejects undeclared asserts) ----
    def declare(self, rel, doc=""):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO schema_rels VALUES(?,?)", (rel, doc))
            self.db.commit()
        return self

    def schema(self):
        return dict(self.db.execute("SELECT rel,doc FROM schema_rels"))

    def _check_schema(self, r):
        if self.get_meta("strict", "0") == "1":
            declared = {x for (x,) in self.db.execute("SELECT rel FROM schema_rels")}
            if declared and r not in declared:
                raise ValueError(f"relation '{r}' is not declared in the schema (strict mode)")

    # ---- materialize / log / status ----
    def to_kg(self):
        kg = KnowledgeGraph()
        for s, r, o in self.triples():
            kg.assert_(s, r, o)
        return kg

    def closure(self):
        """(kg, prov): the KB with every derivable fact materialized, plus provenance.
        Served from cache; recomputed only when facts or rules have changed (pay on write,
        read in milliseconds). Treat the returned kg as read-only."""
        with self.lock:
            rev = self.get_meta("kb_rev", "0")
            if self._cache and self._cache[0] == rev:
                return self._cache[1], self._cache[2]
            kg = self.to_kg()
            if self.get_meta("derived_rev") == rev:             # warm start from the derived table
                prov = {}
                for s, r, o, rule, prem in self.db.execute("SELECT s,r,o,rule,premises FROM derived"):
                    kg.assert_(s, r, o)
                    prov[(s, r, o)] = (rule, [tuple(p) for p in json.loads(prem)])
            else:
                _, prov = forward_chain_prov(kg, self.rules())
                self.db.execute("DELETE FROM derived")
                self.db.executemany("INSERT OR IGNORE INTO derived VALUES(?,?,?,?,?)",
                                    [(s, r, o, rl, json.dumps(prem))
                                     for (s, r, o), (rl, prem) in prov.items()])
                self.db.execute("INSERT OR REPLACE INTO meta VALUES('derived_rev',?)", (rev,))
                self.db.commit()
            self._cache = (rev, kg, prov)
            return kg, prov

    def log(self, kind, detail):
        self.db.execute("INSERT INTO log VALUES(?,?,?)",
                        (time.time(), kind, detail if isinstance(detail, str) else json.dumps(detail)))
        self.db.commit()

    def recent_log(self, n=20):
        return list(self.db.execute("SELECT ts,kind,detail FROM log ORDER BY ts DESC LIMIT ?", (n,)))

    def set_meta(self, key, value):
        with self.lock:
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
                if atom[1] not in GUARDS:
                    rs.add(atom[1].lstrip("~"))
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
