"""
SQLite-backed graph store implementation.

Persists code-relationship graphs (call graphs, dependency graphs),
a cross-file symbol index, and PageRank scores in a single SQLite
database file.  WAL mode is enabled for concurrent reads during indexing.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from aci.core.graph_models import (
    GraphEdge,
    GraphNode,
    SymbolIndexEntry,
    SymbolLocation,
)
from aci.core.graph_store import GraphStoreInterface

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """\
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS graph_nodes (
    symbol_id   TEXT PRIMARY KEY,
    symbol_name TEXT NOT NULL,
    symbol_type TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    language    TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS graph_edges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    inferred    INTEGER NOT NULL DEFAULT 0,
    confidence  REAL NOT NULL DEFAULT 1.0,
    file_path   TEXT NOT NULL DEFAULT '',
    line        INTEGER NOT NULL DEFAULT 0,
    UNIQUE(source_id, target_id, edge_type)
);

CREATE TABLE IF NOT EXISTS pagerank_scores (
    symbol_id   TEXT PRIMARY KEY,
    graph_type  TEXT NOT NULL,
    score       REAL NOT NULL DEFAULT 0.0,
    computed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS symbol_index (
    fqn             TEXT PRIMARY KEY,
    file_path       TEXT NOT NULL,
    start_line      INTEGER NOT NULL,
    end_line        INTEGER NOT NULL,
    symbol_type     TEXT NOT NULL,
    graph_node_id   TEXT NOT NULL,
    summary         TEXT NOT NULL DEFAULT '',
    llm_summary     TEXT NOT NULL DEFAULT '',
    unresolved      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS symbol_references (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fqn         TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    UNIQUE(fqn, file_path, start_line)
);

-- Indexes for hot query paths
CREATE INDEX IF NOT EXISTS idx_edges_source     ON graph_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target     ON graph_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type       ON graph_edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_nodes_file       ON graph_nodes(file_path);
CREATE INDEX IF NOT EXISTS idx_symbol_file      ON symbol_index(file_path);
CREATE INDEX IF NOT EXISTS idx_symbol_refs_fqn  ON symbol_references(fqn);
CREATE INDEX IF NOT EXISTS idx_symbol_refs_file ON symbol_references(file_path);
CREATE INDEX IF NOT EXISTS idx_pagerank_type    ON pagerank_scores(graph_type);
"""


class SQLiteGraphStore(GraphStoreInterface):
    """SQLite-backed graph store.  Data lives at *db_path*."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create tables and indexes.  Idempotent."""
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if self._db_path != ":memory:":
                Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA busy_timeout=5000;")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def upsert_node(self, node: GraphNode) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO graph_nodes "
            "(symbol_id, symbol_name, symbol_type, file_path, start_line, end_line, language) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                node.symbol_id,
                node.symbol_name,
                node.symbol_type,
                node.file_path,
                node.start_line,
                node.end_line,
                node.language,
            ),
        )
        conn.commit()

    def upsert_nodes_batch(self, nodes: list[GraphNode]) -> None:
        if not nodes:
            return
        conn = self._get_conn()
        conn.executemany(
            "INSERT OR REPLACE INTO graph_nodes "
            "(symbol_id, symbol_name, symbol_type, file_path, start_line, end_line, language) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (n.symbol_id, n.symbol_name, n.symbol_type, n.file_path, n.start_line, n.end_line, n.language)
                for n in nodes
            ],
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def upsert_edge(self, edge: GraphEdge) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO graph_edges "
            "(source_id, target_id, edge_type, inferred, confidence, file_path, line) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                edge.source_id,
                edge.target_id,
                edge.edge_type,
                int(edge.inferred),
                edge.confidence,
                edge.file_path,
                edge.line,
            ),
        )
        conn.commit()

    def upsert_edges_batch(self, edges: list[GraphEdge]) -> None:
        if not edges:
            return
        conn = self._get_conn()
        conn.executemany(
            "INSERT OR REPLACE INTO graph_edges "
            "(source_id, target_id, edge_type, inferred, confidence, file_path, line) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (e.source_id, e.target_id, e.edge_type, int(e.inferred), e.confidence, e.file_path, e.line)
                for e in edges
            ],
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_by_file(self, file_path: str) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM graph_edges WHERE file_path = ?", (file_path,))
        conn.execute("DELETE FROM graph_nodes WHERE file_path = ?", (file_path,))
        conn.execute("DELETE FROM symbol_index WHERE file_path = ?", (file_path,))
        conn.execute("DELETE FROM symbol_references WHERE file_path = ?", (file_path,))
        conn.commit()

    # ------------------------------------------------------------------
    # Traversal / query  (recursive CTE)
    # ------------------------------------------------------------------

    # Pre-built SQL templates keyed by (direction, include_inferred).
    # Using static strings avoids f-string interpolation that triggers S608.

    _NEIGHBOR_CALLEES = (  # noqa: S608
        "WITH RECURSIVE traversal(symbol_id, depth) AS ("
        "  SELECT :start, 0"
        "  UNION ALL"
        "  SELECT e.target_id, t.depth + 1"
        "  FROM graph_edges e"
        "  JOIN traversal t ON e.source_id = t.symbol_id"
        "  WHERE t.depth < :max_depth"
        ") "
        "SELECT DISTINCT n.* FROM traversal tr "
        "JOIN graph_nodes n ON n.symbol_id = tr.symbol_id "
        "WHERE tr.symbol_id != :start"
    )

    _NEIGHBOR_CALLEES_NO_INFERRED = (  # noqa: S608
        "WITH RECURSIVE traversal(symbol_id, depth) AS ("
        "  SELECT :start, 0"
        "  UNION ALL"
        "  SELECT e.target_id, t.depth + 1"
        "  FROM graph_edges e"
        "  JOIN traversal t ON e.source_id = t.symbol_id"
        "  WHERE t.depth < :max_depth AND e.inferred = 0"
        ") "
        "SELECT DISTINCT n.* FROM traversal tr "
        "JOIN graph_nodes n ON n.symbol_id = tr.symbol_id "
        "WHERE tr.symbol_id != :start"
    )

    _NEIGHBOR_CALLERS = (  # noqa: S608
        "WITH RECURSIVE traversal(symbol_id, depth) AS ("
        "  SELECT :start, 0"
        "  UNION ALL"
        "  SELECT e.source_id, t.depth + 1"
        "  FROM graph_edges e"
        "  JOIN traversal t ON e.target_id = t.symbol_id"
        "  WHERE t.depth < :max_depth"
        ") "
        "SELECT DISTINCT n.* FROM traversal tr "
        "JOIN graph_nodes n ON n.symbol_id = tr.symbol_id "
        "WHERE tr.symbol_id != :start"
    )

    _NEIGHBOR_CALLERS_NO_INFERRED = (  # noqa: S608
        "WITH RECURSIVE traversal(symbol_id, depth) AS ("
        "  SELECT :start, 0"
        "  UNION ALL"
        "  SELECT e.source_id, t.depth + 1"
        "  FROM graph_edges e"
        "  JOIN traversal t ON e.target_id = t.symbol_id"
        "  WHERE t.depth < :max_depth AND e.inferred = 0"
        ") "
        "SELECT DISTINCT n.* FROM traversal tr "
        "JOIN graph_nodes n ON n.symbol_id = tr.symbol_id "
        "WHERE tr.symbol_id != :start"
    )

    _EDGES_CALLEES = (  # noqa: S608
        "WITH RECURSIVE traversal(symbol_id, depth) AS ("
        "  SELECT :start, 0"
        "  UNION ALL"
        "  SELECT e.target_id, t.depth + 1"
        "  FROM graph_edges e"
        "  JOIN traversal t ON e.source_id = t.symbol_id"
        "  WHERE t.depth < :max_depth"
        ") "
        "SELECT DISTINCT e2.* FROM graph_edges e2 "
        "JOIN traversal t1 ON e2.source_id = t1.symbol_id "
        "JOIN traversal t2 ON e2.target_id = t2.symbol_id"
    )

    _EDGES_CALLEES_NO_INFERRED = (  # noqa: S608
        "WITH RECURSIVE traversal(symbol_id, depth) AS ("
        "  SELECT :start, 0"
        "  UNION ALL"
        "  SELECT e.target_id, t.depth + 1"
        "  FROM graph_edges e"
        "  JOIN traversal t ON e.source_id = t.symbol_id"
        "  WHERE t.depth < :max_depth AND e.inferred = 0"
        ") "
        "SELECT DISTINCT e2.* FROM graph_edges e2 "
        "JOIN traversal t1 ON e2.source_id = t1.symbol_id "
        "JOIN traversal t2 ON e2.target_id = t2.symbol_id"
        " WHERE e2.inferred = 0"
    )

    _EDGES_CALLERS = (  # noqa: S608
        "WITH RECURSIVE traversal(symbol_id, depth) AS ("
        "  SELECT :start, 0"
        "  UNION ALL"
        "  SELECT e.source_id, t.depth + 1"
        "  FROM graph_edges e"
        "  JOIN traversal t ON e.target_id = t.symbol_id"
        "  WHERE t.depth < :max_depth"
        ") "
        "SELECT DISTINCT e2.* FROM graph_edges e2 "
        "JOIN traversal t1 ON e2.source_id = t1.symbol_id "
        "JOIN traversal t2 ON e2.target_id = t2.symbol_id"
    )

    _EDGES_CALLERS_NO_INFERRED = (  # noqa: S608
        "WITH RECURSIVE traversal(symbol_id, depth) AS ("
        "  SELECT :start, 0"
        "  UNION ALL"
        "  SELECT e.source_id, t.depth + 1"
        "  FROM graph_edges e"
        "  JOIN traversal t ON e.target_id = t.symbol_id"
        "  WHERE t.depth < :max_depth AND e.inferred = 0"
        ") "
        "SELECT DISTINCT e2.* FROM graph_edges e2 "
        "JOIN traversal t1 ON e2.source_id = t1.symbol_id "
        "JOIN traversal t2 ON e2.target_id = t2.symbol_id"
        " WHERE e2.inferred = 0"
    )

    def get_neighbors(
        self,
        symbol_id: str,
        direction: str,
        depth: int = 1,
        include_inferred: bool = True,
    ) -> list[GraphNode]:
        conn = self._get_conn()
        if direction == "callees":
            sql = self._NEIGHBOR_CALLEES if include_inferred else self._NEIGHBOR_CALLEES_NO_INFERRED
        else:
            sql = self._NEIGHBOR_CALLERS if include_inferred else self._NEIGHBOR_CALLERS_NO_INFERRED
        rows = conn.execute(sql, {"start": symbol_id, "max_depth": depth}).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_edges(
        self,
        symbol_id: str,
        direction: str,
        depth: int = 1,
        include_inferred: bool = True,
    ) -> list[GraphEdge]:
        conn = self._get_conn()
        if direction == "callees":
            sql = self._EDGES_CALLEES if include_inferred else self._EDGES_CALLEES_NO_INFERRED
        else:
            sql = self._EDGES_CALLERS if include_inferred else self._EDGES_CALLERS_NO_INFERRED
        rows = conn.execute(sql, {"start": symbol_id, "max_depth": depth}).fetchall()
        return [self._row_to_edge(r) for r in rows]

    # ------------------------------------------------------------------
    # PageRank
    # ------------------------------------------------------------------

    def get_pagerank(self, symbol_id: str, graph_type: str = "call") -> float:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT score FROM pagerank_scores WHERE symbol_id = ? AND graph_type = ?",
            (symbol_id, graph_type),
        ).fetchone()
        return float(row["score"]) if row else 0.0

    def store_pagerank_scores(
        self, scores: dict[str, float], graph_type: str
    ) -> None:
        if not scores:
            return
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.executemany(
            "INSERT OR REPLACE INTO pagerank_scores "
            "(symbol_id, graph_type, score, computed_at) VALUES (?, ?, ?, ?)",
            [(sid, graph_type, score, now) for sid, score in scores.items()],
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Symbol / module queries
    # ------------------------------------------------------------------

    def query_symbol(self, symbol_id: str) -> GraphNode | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM graph_nodes WHERE symbol_id = ?", (symbol_id,)
        ).fetchone()
        return self._row_to_node(row) if row else None

    def query_module(self, file_path: str) -> dict:
        conn = self._get_conn()
        nodes = conn.execute(
            "SELECT * FROM graph_nodes WHERE file_path = ?", (file_path,)
        ).fetchall()
        # Edges originating from this file
        edges = conn.execute(
            "SELECT * FROM graph_edges WHERE file_path = ?", (file_path,)
        ).fetchall()
        return {
            "nodes": [self._row_to_node(r) for r in nodes],
            "edges": [self._row_to_edge(r) for r in edges],
        }

    # ------------------------------------------------------------------
    # Serialization (export / import)
    # ------------------------------------------------------------------

    def export_json(self, path: str) -> None:
        conn = self._get_conn()
        nodes = conn.execute("SELECT * FROM graph_nodes").fetchall()
        edges = conn.execute("SELECT * FROM graph_edges").fetchall()
        pr = conn.execute("SELECT * FROM pagerank_scores").fetchall()

        data = {
            "schema_version": "1.0",
            "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "nodes": [
                {
                    "symbol_id": r["symbol_id"],
                    "symbol_name": r["symbol_name"],
                    "symbol_type": r["symbol_type"],
                    "file_path": r["file_path"],
                    "start_line": r["start_line"],
                    "end_line": r["end_line"],
                    "language": r["language"],
                }
                for r in nodes
            ],
            "edges": [
                {
                    "source_id": r["source_id"],
                    "target_id": r["target_id"],
                    "edge_type": r["edge_type"],
                    "inferred": bool(r["inferred"]),
                    "confidence": r["confidence"],
                    "file_path": r["file_path"],
                    "line": r["line"],
                }
                for r in edges
            ],
            "pagerank_scores": [
                {
                    "symbol_id": r["symbol_id"],
                    "graph_type": r["graph_type"],
                    "score": r["score"],
                }
                for r in pr
            ],
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def import_json(self, path: str, mode: str) -> None:
        raw = Path(path).read_text(encoding="utf-8")
        data = json.loads(raw)
        conn = self._get_conn()

        if mode == "replace":
            conn.execute("BEGIN")
            try:
                conn.execute("DELETE FROM graph_edges")
                conn.execute("DELETE FROM graph_nodes")
                conn.execute("DELETE FROM pagerank_scores")
                self._bulk_insert_from_dict(conn, data)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        elif mode == "merge":
            self._bulk_insert_from_dict(conn, data)
            conn.commit()
        else:
            raise ValueError(f"Unknown import mode: {mode!r}. Use 'replace' or 'merge'.")

    def _bulk_insert_from_dict(self, conn: sqlite3.Connection, data: dict) -> None:
        for n in data.get("nodes", []):
            conn.execute(
                "INSERT OR REPLACE INTO graph_nodes "
                "(symbol_id, symbol_name, symbol_type, file_path, start_line, end_line, language) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    n["symbol_id"],
                    n["symbol_name"],
                    n["symbol_type"],
                    n["file_path"],
                    n["start_line"],
                    n["end_line"],
                    n.get("language", ""),
                ),
            )
        for e in data.get("edges", []):
            conn.execute(
                "INSERT OR REPLACE INTO graph_edges "
                "(source_id, target_id, edge_type, inferred, confidence, file_path, line) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    e["source_id"],
                    e["target_id"],
                    e["edge_type"],
                    int(e.get("inferred", False)),
                    e.get("confidence", 1.0),
                    e.get("file_path", ""),
                    e.get("line", 0),
                ),
            )
        for p in data.get("pagerank_scores", []):
            conn.execute(
                "INSERT OR REPLACE INTO pagerank_scores "
                "(symbol_id, graph_type, score, computed_at) VALUES (?, ?, ?, ?)",
                (
                    p["symbol_id"],
                    p["graph_type"],
                    p["score"],
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    # ------------------------------------------------------------------
    # Bulk accessors
    # ------------------------------------------------------------------

    def get_all_edges(self, graph_type: str | None = None) -> list[GraphEdge]:
        conn = self._get_conn()
        if graph_type is not None:
            rows = conn.execute(
                "SELECT * FROM graph_edges WHERE edge_type = ?", (graph_type,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM graph_edges").fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_all_nodes(self) -> list[GraphNode]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM graph_nodes").fetchall()
        return [self._row_to_node(r) for r in rows]

    # ------------------------------------------------------------------
    # Symbol index operations
    # ------------------------------------------------------------------

    def upsert_symbol(self, entry: SymbolIndexEntry) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO symbol_index "
            "(fqn, file_path, start_line, end_line, symbol_type, graph_node_id, "
            "summary, llm_summary, unresolved) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.fqn,
                entry.definition.file_path,
                entry.definition.start_line,
                entry.definition.end_line,
                self._infer_symbol_type(entry),
                entry.graph_node_id,
                entry.summary,
                entry.llm_summary,
                int(entry.unresolved),
            ),
        )
        # Upsert references
        self._upsert_references(conn, entry)
        conn.commit()

    def upsert_symbols_batch(self, entries: list[SymbolIndexEntry]) -> None:
        if not entries:
            return
        conn = self._get_conn()
        for entry in entries:
            conn.execute(
                "INSERT OR REPLACE INTO symbol_index "
                "(fqn, file_path, start_line, end_line, symbol_type, graph_node_id, "
                "summary, llm_summary, unresolved) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.fqn,
                    entry.definition.file_path,
                    entry.definition.start_line,
                    entry.definition.end_line,
                    self._infer_symbol_type(entry),
                    entry.graph_node_id,
                    entry.summary,
                    entry.llm_summary,
                    int(entry.unresolved),
                ),
            )
            self._upsert_references(conn, entry)
        conn.commit()

    def lookup_symbol(self, fqn: str) -> SymbolIndexEntry | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM symbol_index WHERE fqn = ?", (fqn,)
        ).fetchone()
        if row is None:
            return None
        refs = conn.execute(
            "SELECT * FROM symbol_references WHERE fqn = ?", (fqn,)
        ).fetchall()
        return self._row_to_symbol_entry(row, refs)

    def lookup_symbols_by_name(self, short_name: str) -> list[SymbolIndexEntry]:
        conn = self._get_conn()
        # Match FQNs that end with .<short_name> or equal short_name exactly
        rows = conn.execute(
            "SELECT * FROM symbol_index WHERE fqn = ? OR fqn LIKE ?",
            (short_name, f"%.{short_name}"),
        ).fetchall()
        results: list[SymbolIndexEntry] = []
        for row in rows:
            refs = conn.execute(
                "SELECT * FROM symbol_references WHERE fqn = ?", (row["fqn"],)
            ).fetchall()
            results.append(self._row_to_symbol_entry(row, refs))
        return results

    def get_symbols_in_file(self, file_path: str) -> list[SymbolIndexEntry]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM symbol_index WHERE file_path = ?", (file_path,)
        ).fetchall()
        results: list[SymbolIndexEntry] = []
        for row in rows:
            refs = conn.execute(
                "SELECT * FROM symbol_references WHERE fqn = ?", (row["fqn"],)
            ).fetchall()
            results.append(self._row_to_symbol_entry(row, refs))
        return results

    def delete_symbols_by_file(self, file_path: str) -> None:
        conn = self._get_conn()
        # Delete references for symbols defined in this file
        fqns = conn.execute(
            "SELECT fqn FROM symbol_index WHERE file_path = ?", (file_path,)
        ).fetchall()
        for row in fqns:
            conn.execute("DELETE FROM symbol_references WHERE fqn = ?", (row["fqn"],))
        conn.execute("DELETE FROM symbol_index WHERE file_path = ?", (file_path,))
        conn.commit()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> GraphNode:
        return GraphNode(
            symbol_id=row["symbol_id"],
            symbol_name=row["symbol_name"],
            symbol_type=row["symbol_type"],
            file_path=row["file_path"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            language=row["language"],
        )

    @staticmethod
    def _row_to_edge(row: sqlite3.Row) -> GraphEdge:
        return GraphEdge(
            source_id=row["source_id"],
            target_id=row["target_id"],
            edge_type=row["edge_type"],
            inferred=bool(row["inferred"]),
            confidence=row["confidence"],
            file_path=row["file_path"],
            line=row["line"],
        )

    @staticmethod
    def _row_to_symbol_entry(
        row: sqlite3.Row, ref_rows: list[sqlite3.Row]
    ) -> SymbolIndexEntry:
        return SymbolIndexEntry(
            fqn=row["fqn"],
            definition=SymbolLocation(
                file_path=row["file_path"],
                start_line=row["start_line"],
                end_line=row["end_line"],
            ),
            references=[
                SymbolLocation(
                    file_path=r["file_path"],
                    start_line=r["start_line"],
                    end_line=r["end_line"],
                )
                for r in ref_rows
            ],
            graph_node_id=row["graph_node_id"],
            summary=row["summary"],
            llm_summary=row["llm_summary"],
            unresolved=bool(row["unresolved"]),
        )

    @staticmethod
    def _infer_symbol_type(entry: SymbolIndexEntry) -> str:
        """Derive a symbol_type string from the entry's graph_node_id or fqn."""
        # If the entry has a graph_node_id that matches a known node, the
        # caller should have set it.  Fall back to "unknown".
        return "unknown"

    @staticmethod
    def _upsert_references(
        conn: sqlite3.Connection, entry: SymbolIndexEntry
    ) -> None:
        for ref in entry.references:
            conn.execute(
                "INSERT OR REPLACE INTO symbol_references "
                "(fqn, file_path, start_line, end_line) VALUES (?, ?, ?, ?)",
                (entry.fqn, ref.file_path, ref.start_line, ref.end_line),
            )
