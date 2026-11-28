"""Domain 1 (Fictional Python Frameworks: veloxdb & nexflow) + Standard Python Sandbox Suites."""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Seed 42: veloxdb — Embedded Columnar Micro-Store
# ---------------------------------------------------------------------------

VELOXDB_MOCK_SRC = '''"""Mock implementation of veloxdb v2.4 enforcing strict API & state rules."""

VELOX_PORT = 7419
MAX_BATCH = 256


class DeprecationError(RuntimeError):
    pass


class StateViolationError(RuntimeError):
    pass


class BatchLimitError(ValueError):
    pass


class CursorFrame:
    """Lazy query result wrapper; direct iteration yields (row_id, dict) tuples."""

    def __init__(self, rows):
        self._rows = [dict(r) for r in rows]

    def __iter__(self):
        for idx, r in enumerate(self._rows):
            yield (idx, dict(r))

    def __len__(self):
        return len(self._rows)

    def materialize(self):
        return [dict(r) for r in self._rows]


_STORE = {}


class VaultEngine:
    def __init__(self, path: str, mode: str = "rw", page_kb: int = 16):
        if mode not in ("rw", "ro"):
            raise ValueError(f"Unsupported mode: {mode}")
        self.path = path
        self.mode = mode
        self.page_kb = int(page_kb)
        self.mounted = False
        self.compacted = False
        self.wal_syncs = 0
        if path not in _STORE:
            _STORE[path] = {}

    def mount(self):
        self.mounted = True
        return self

    def unmount(self):
        self.mounted = False

    def insert_rows(self, *args, **kwargs):
        raise DeprecationError("insert_rows() was removed in v2.4; use write_batch(records, table_name)")

    def write_batch(self, records, table_name: str, *, sync_wal: bool = True, **kwargs):
        if "flush_immediately" in kwargs:
            raise DeprecationError("flush_immediately= was removed; use keyword-only sync_wal=")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")
        if not self.mounted:
            raise StateViolationError("VaultEngine must be mounted via .mount() before write_batch()")
        if self.mode != "rw":
            raise StateViolationError("Cannot write_batch() when mounted in 'ro' mode")
        if not isinstance(records, (list, tuple)) or not isinstance(table_name, str):
            raise TypeError("Argument order is write_batch(records: list[dict], table_name: str)")
        if len(records) > MAX_BATCH:
            raise BatchLimitError(f"Batch size {len(records)} exceeds MAX_BATCH={MAX_BATCH}")
        tbl = _STORE[self.path].setdefault(table_name, [])
        for r in records:
            tbl.append(dict(r))
        if sync_wal:
            self.wal_syncs += 1
        return len(records)

    def scan(self, table_name: str, where=None, limit: int = 50):
        if not self.mounted:
            raise StateViolationError("VaultEngine must be mounted via .mount() before scan()")
        rows = _STORE[self.path].get(table_name, [])
        if where:
            rows = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
        return CursorFrame(rows[: int(limit)])


def fold_metric(cursor: CursorFrame, field: str, op: str = "sum_sq") -> float:
    if not isinstance(cursor, CursorFrame):
        raise TypeError("fold_metric() requires a CursorFrame object, not a materialized list")
    vals = [float(r[field]) for _, r in cursor if field in r]
    if op == "sum_sq":
        return float(sum(v * v for v in vals))
    if op == "sum":
        return float(sum(vals))
    if op == "mean":
        return float(sum(vals) / len(vals)) if vals else 0.0
    raise ValueError(f"Unknown op={op}")


def compact_segments(engine: VaultEngine, ratio_floor: float = 0.35) -> dict:
    if not isinstance(engine, VaultEngine):
        raise TypeError("compact_segments() expects a VaultEngine instance")
    if engine.mounted:
        raise StateViolationError("Engine must be unmounted via .unmount() before compact_segments()")
    engine.compacted = True
    tables = _STORE.get(engine.path, {})
    return {
        "path": engine.path,
        "ratio_floor": float(ratio_floor),
        "tables": len(tables),
        "rows_total": sum(len(v) for v in tables.values()),
        "page_kb": engine.page_kb,
    }
'''

VELOXDB_DOC = """# veloxdb v2.4 Architecture & Developer Field Manual

## 1. Design Philosophy and Storage Geometry

`veloxdb` is a zero-dependency embedded columnar micro-store designed for telemetry buffers, sensor rings, and deterministic local state snapshots on constrained edge nodes. Unlike traditional client-server relational engines, `veloxdb` embeds its page cache directly inside the host Python process and binds TCP control telemetry only when the optional diagnostic inspector is attached on port `7419` (`veloxdb.VELOX_PORT = 7419`). By default, page allocations are aligned to 16 KiB boundaries (`page_kb=16`), though ingestion-heavy workloads running over solid-state NVMe partitions routinely allocate 32 KiB or 64 KiB pages to reduce index churn during burst appends.

At the center of the library sits `veloxdb.VaultEngine(path, mode="rw", page_kb=16)`. Every engine handle begins in an idle, unmounted state immediately after instantiation. No read, write, or schema lookup may occur until the caller explicitly activates the page table with `engine.mount()`. Conversely, lifecycle cleanup relies on `engine.unmount()`, which flushes dirty header descriptors back to the backing store. Attempting to issue I/O operations while unmounted raises `veloxdb.StateViolationError`. Similarly, opening a store with `mode="ro"` locks out mutations at the engine boundary and raises `StateViolationError` if a caller attempts to insert records.

## 2. Batch Ingestion, Argument Ordering, and v2.4 Breaking Changes

Prior to release 2.4, developers ingested rows through a convenience method named `insert_rows(table_name, records, flush_immediately=True)`. Because telemetry benchmarks showed repeated confusion between positional table names and streaming iterators, release 2.4 completely removed `insert_rows()` and banned the `flush_immediately` keyword argument. Calling `insert_rows()` or passing `flush_immediately=` in v2.4 immediately triggers `veloxdb.DeprecationError`.

All modern writes go through `VaultEngine.write_batch(records, table_name, *, sync_wal=True)`. Pay careful attention to two strict constraints in this signature:
1. **Reversed Positional Order**: `records` (a list or tuple of dictionaries) comes **first**, followed by `table_name` (a string) **second**. Passing `(table_name, records)` raises `TypeError`.
2. **Keyword-Only WAL Flag**: `sync_wal` is a keyword-only boolean defaulting to `True`. When high-throughput ingestion pipelines write synthetic backfills where crash durability can be deferred until `unmount()`, callers pass `sync_wal=False`.

To bound peak memory per write-ahead log transaction, a single `write_batch` invocation accepts at most `256` records (`veloxdb.MAX_BATCH = 256`). Supplying 257 or more dictionaries in one call raises `veloxdb.BatchLimitError`. Production loaders ingesting arbitrary collections must slice payloads into chunks of at most `veloxdb.MAX_BATCH` items before invoking `write_batch(chunk, table_name)`.

## 3. Query Cursors, Materialization Traps, and Folded Metrics

Queries are executed via `VaultEngine.scan(table_name, where=None, limit=50)`. Notice that `scan()` does **not** return a plain Python list of dictionaries. Instead, it returns a lazy `CursorFrame` view wrapping the matched slice (bounded by `limit`, which defaults to `50`). Iterating over a `CursorFrame` directly with `for item in cursor` yields two-element `(row_id, row_dict)` tuples rather than plain row dictionaries—a deliberate design choice so streaming reducers can inspect internal slot offsets without copying payloads. Whenever downstream code needs a standard `list[dict]`, it must call `cursor.materialize()`.

For numeric aggregation across query results, `veloxdb` exposes `veloxdb.fold_metric(cursor, field, op="sum_sq")`. Three subtleties frequently trip up new contributors:
- `fold_metric` expects the unmaterialized `CursorFrame` returned by `scan()`, **not** the `list[dict]` returned by `.materialize()`. Passing a plain Python list raises `TypeError`.
- The default reduction operator `op` is `"sum_sq"` (the sum of squared numeric values in `field`, useful for RMS energy diagnostics), rather than `"sum"`. Other supported modes are `"sum"` and `"mean"`.
- Because `CursorFrame` only reflects rows already sliced by `scan(..., limit=...)`, callers migrating large tables should specify an explicit `limit` large enough to capture all stored rows (for example `limit=1000`).

## 4. Offline Segment Compaction and Lifecycle Hygiene

Over repeated `write_batch` cycles, tombstones and split page fragments accumulate across table segments. Maintenance scripts reclaim space by calling the module-level function `veloxdb.compact_segments(engine, ratio_floor=0.35)`. Unlike `scan()` or `write_batch()`, segment compaction acquires an exclusive lock across the backing directory and therefore **forbids** active mounts: `compact_segments(engine, ...)` must be called strictly **after** `engine.unmount()`. Calling `compact_segments` while `engine.mounted` is `True` raises `veloxdb.StateViolationError`.

When successful, `compact_segments` returns a diagnostic dictionary containing keys `"path"`, `"ratio_floor"`, `"tables"`, `"rows_total"`, and `"page_kb"`. The default `ratio_floor` threshold is `0.35`, triggering segment merging whenever dead bytes exceed 35 percent of allocated pages.

## 5. Production Migration Notes and Operational Troubleshooting

Field post-mortems from the v2.2-to-v2.4 transition highlight three recurring failure modes in automated data pipelines:
First, scripts wrapped around `VaultEngine(path, mode="ro")` occasionally leave handles mounted if an upstream filter raises an exception before cleanup. Wrapping query logic in `try ... finally: engine.unmount()` guarantees file descriptors and mount locks are released before subsequent compaction jobs run on the same `path`.
Second, legacy ETL helpers migrating rows from a staging table to a production table must avoid calling `insert_rows`. Instead, open a `"rw"` engine, mount once, read `cursor = engine.scan(src_table, limit=1000)` followed by `rows = cursor.materialize()`, write `rows` into `dst_table` via `engine.write_batch(rows, dst_table)` (chunking if needed), and unmount cleanly.
Third, telemetry aggregators computing RMS norms sometimes call `cursor.materialize()` too early and then pass the resulting list into `fold_metric`. Keeping the `CursorFrame` object intact lets callers pass `cursor` directly to `veloxdb.fold_metric(cursor, field)` and separately materialize `cursor.materialize()` when returning JSON-serializable records to the caller.

### Appendix: Exception Hierarchy Summary
- `veloxdb.StateViolationError`: Raised when performing mounted-only I/O (`write_batch`, `scan`) on an unmounted handle, writing on an `"ro"` handle, or invoking `compact_segments()` while the engine is still mounted.
- `veloxdb.DeprecationError`: Raised whenever retired v2.2 symbols (`VaultEngine.insert_rows`, `flush_immediately=`) are invoked.
- `veloxdb.BatchLimitError`: Raised when a single `write_batch` call receives more than `MAX_BATCH` (`256`) dictionaries.
"""

# ---------------------------------------------------------------------------
# Seed 1337: nexflow — DAG Stream Processing Runtime
# ---------------------------------------------------------------------------

NEXFLOW_MOCK_SRC = '''"""Mock implementation of nexflow v3.1 enforcing strict DAG & state rules."""

NEXFLOW_MAGIC = "NFX-409"
MAX_RETRIES = 3


class LegacyRouteError(RuntimeError):
    pass


class UnarmedGraphError(RuntimeError):
    pass


class StageTicket:
    def __init__(self, stage_tag: str, stage_fn, priority: int):
        self.stage_tag = stage_tag
        self.stage_fn = stage_fn
        self.priority = int(priority)
        self.sealed = False

    def seal(self):
        self.sealed = True
        return self


class DrainBundle:
    """Ordered output wrapper; direct iteration yields (seq_no, value) tuples."""

    def __init__(self, pairs):
        self._pairs = list(pairs)

    def __iter__(self):
        for seq_no, val in self._pairs:
            yield (seq_no, val)

    def __len__(self):
        return len(self._pairs)

    def as_values(self):
        return [val for _, val in self._pairs]


class PulseGraph:
    def __init__(self, name: str, workers: int = 2, strict: bool = True):
        self.name = name
        self.workers = int(workers)
        self.strict = bool(strict)
        self.armed = False
        self.stages = []
        self._lanes = {}

    def arm(self):
        if self.strict:
            for st in self.stages:
                if not st.sealed:
                    raise UnarmedGraphError(f"Stage {st.stage_tag!r} must be sealed via .seal() before arm()")
        self.armed = True
        return self

    def disarm(self):
        self.armed = False

    def send(self, *args, **kwargs):
        raise LegacyRouteError("send() was removed in v3.1; use dispatch(payload, lane_id)")

    def dispatch(self, payload, lane_id: str, *, buffered: bool = False, **kwargs):
        if "async_mode" in kwargs:
            raise LegacyRouteError("async_mode= was removed; use keyword-only buffered=")
        if kwargs:
            raise TypeError(f"Unexpected kwargs: {list(kwargs.keys())}")
        if not self.armed:
            raise UnarmedGraphError("PulseGraph must be armed via .arm() before dispatch()")
        if not isinstance(lane_id, str):
            raise TypeError("Argument order is dispatch(payload, lane_id: str)")
        val = payload
        for st in sorted(self.stages, key=lambda s: s.priority):
            val = st.stage_fn(val)
        lane = self._lanes.setdefault(lane_id, [])
        seq_no = len(lane)
        lane.append((seq_no, val))
        return seq_no

    def drain(self, lane_id: str, keep_order: bool = True):
        if not self.armed:
            raise UnarmedGraphError("PulseGraph must be armed via .arm() before drain()")
        pairs = list(self._lanes.pop(lane_id, []))
        if not keep_order:
            pairs = list(reversed(pairs))
        return DrainBundle(pairs)


def bind_stage(graph: PulseGraph, stage_fn, stage_tag: str, priority: int = 10) -> StageTicket:
    if not isinstance(graph, PulseGraph) or not callable(stage_fn) or not isinstance(stage_tag, str):
        raise TypeError("Argument order is bind_stage(graph: PulseGraph, stage_fn, stage_tag: str, priority=10)")
    if graph.armed:
        raise UnarmedGraphError("Cannot bind_stage() while graph is armed")
    ticket = StageTicket(stage_tag=stage_tag, stage_fn=stage_fn, priority=priority)
    graph.stages.append(ticket)
    return ticket
'''

NEXFLOW_DOC = """# nexflow v3.1 Stream DAG Runtime Reference

## 1. Runtime Model and Control Constants

`nexflow` is a deterministic single-process DAG stream engine designed for low-jitter transformation pipelines and reproducible telemetry routing. Every frame packet carries the internal protocol signature `NFX-409` (`nexflow.NEXFLOW_MAGIC = "NFX-409"`), and transient transport handlers cap retry attempts at `3` (`nexflow.MAX_RETRIES = 3`).

Pipeline topologies are constructed around `nexflow.PulseGraph(name, workers=2, strict=True)`. By default, a new graph starts in an unarmed configuration (`workers=2`, `strict=True`) while stages are attached. Before any payload can enter or exit the DAG, callers must transition the runtime by calling `graph.arm()`. When `strict=True`, `arm()` enforces that every registered stage ticket has already been finalized with `.seal()`. If any attached stage remains unsealed, or if `dispatch()` or `drain()` is called prior to `arm()`, the runtime raises `nexflow.UnarmedGraphError`. Calling `graph.disarm()` pauses execution and returns the graph to configuration mode.

## 2. Stage Binding Order and Sealing Protocol

Transform operators are attached using the module-level function `nexflow.bind_stage(graph, stage_fn, stage_tag, priority=10)`. Several strict signature conventions apply:
1. **Positional Order**: The arguments must appear as `(graph, stage_fn, stage_tag)` — the `PulseGraph` first, the callable transform `stage_fn` second, and the string identifier `stage_tag` third. Swapping `stage_tag` and `stage_fn` raises `TypeError`.
2. **Priority Scheduling**: Stages execute in ascending `priority` order (`priority=10` by default, so a stage registered with `priority=5` runs before `priority=10`).
3. **Pre-Arm Binding & Ticket Sealing**: `bind_stage()` cannot be called once `graph.arm()` is active; doing so raises `UnarmedGraphError`. Furthermore, `bind_stage(...)` returns a `StageTicket` instance. Callers must invoke `ticket.seal()` on each returned ticket before calling `graph.arm()`.

## 3. Dispatching Payloads and v3.1 Deprecation Traps

Legacy `nexflow` v2 scripts routed events via `graph.send(lane_id, payload, async_mode=True)`. In v3.1, both the `.send()` method and the `async_mode=` keyword argument were permanently retired; invoking either immediately raises `nexflow.LegacyRouteError`.

Modern ingestion uses `PulseGraph.dispatch(payload, lane_id, *, buffered=False)`. Notice two critical rules:
- **Payload Before Lane**: `dispatch()` accepts `payload` **first** and `lane_id` (a string) **second** — the exact opposite of the deprecated `send(lane_id, payload)` signature. Passing a non-string second argument raises `TypeError`.
- **Keyword-Only Buffering Flag**: `buffered` is keyword-only and defaults to `False`.

Each `dispatch(payload, lane_id)` call runs all bound stages sorted by ascending `priority` and appends `(seq_no, transformed_value)` to the specified `lane_id`, returning the zero-based integer `seq_no`.

## 4. Draining Lanes Without Falling Into the Tuple Wrapper Trap

To retrieve processed items from a lane, callers invoke `graph.drain(lane_id, keep_order=True)`. Note that `drain()` clears that lane's buffer and returns a `DrainBundle` wrapper rather than a bare Python list. Iterating over a `DrainBundle` directly yields `(seq_no, value)` tuples. Whenever user code expects a plain list of transformed output values, it **must** call `bundle.as_values()`. Passing `keep_order=False` reverses the drained sequence order inside the bundle.

## 5. Multi-Stage Composition Patterns and Troubleshooting

Engineers upgrading from v2.x most frequently encounter three runtime exceptions during integration tests:
First, forgetting `ticket.seal()` after `nexflow.bind_stage(g, fn, "norm")` triggers `UnarmedGraphError` inside `g.arm()` whenever `strict=True`. Calling `nexflow.bind_stage(...).seal()` inline or sealing each ticket explicitly resolves the error cleanly.
Second, passing `lane_id` before `payload` in `graph.dispatch` causes `TypeError` because `lane_id` must be the second positional argument (`str`). Likewise, leftover references to `graph.send(...)` or `async_mode=` fail loudly with `LegacyRouteError`.
Third, tests asserting equality against `[10, 20, 30]` fail if callers return `graph.drain(lane_id)` directly instead of `graph.drain(lane_id).as_values()`, because iterating a `DrainBundle` yields `(0, 10), (1, 20), (2, 30)` tuples instead of scalar outputs.

## 6. Concurrency Semantics, Re-Arming, and Performance Benchmarks

Although `PulseGraph(name, workers=2, strict=True)` records a `workers` hint for multi-lane scheduler diagnostics, single-process execution preserves deterministic sequence numbering (`seq_no` starting at `0` per lane) across every `dispatch()` call. When a topology needs dynamic mid-run updates—such as appending a post-processing square or audit stage after an initial calibration batch—callers must first pause the graph via `graph.disarm()`, register and seal the additional `StageTicket` with `bind_stage(..., priority=...)` + `.seal()`, and then invoke `graph.arm()` again before resuming `dispatch()` calls onto existing or new lanes. Attempting to call `bind_stage()` on an already armed graph always fails fast with `UnarmedGraphError` to guarantee stage ordering cannot mutate mid-flight.

Benchmarks on industrial edge telemetry pipelines demonstrate why strict stage sealing and explicit `.as_values()` extraction were standardized in v3.1. Across 50,000 frame dispatches tagged with `NEXFLOW_MAGIC` (`"NFX-409"`), buffered dispatch (`buffered=True`) reduced Python wrapper allocation overhead by 41% compared with v2.x `send(..., async_mode=True)`, while priority-sorted stage execution (`priority=5` before `priority=20`) eliminated non-deterministic lane race bugs observed in earlier custom schedulers. When transient downstream transport sinks drop packets, adapters should consult `nexflow.MAX_RETRIES` (`3`) rather than hardcoding magic retry bounds.

### Appendix: Quick Checklist for Valid v3.1 Pipelines
1. Instantiate `g = nexflow.PulseGraph(name, workers=2, strict=True)`.
2. Attach stages with `t = nexflow.bind_stage(g, fn, tag, priority=...)` and call `t.seal()` for every stage.
3. Activate via `g.arm()`.
4. Push records with `g.dispatch(payload, lane_id, buffered=...)` — never `g.send()`.
5. Collect outputs with `g.drain(lane_id).as_values()`, and optionally call `g.disarm()` after teardown.
"""


def get_domain1_data(seed: int) -> Dict[str, object]:
    if seed == 42:
        return _veloxdb_bundle()
    if seed == 1337:
        return _nexflow_bundle()
    raise ValueError(f"Unsupported Domain 1 seed: {seed}")


def _veloxdb_bundle() -> Dict[str, object]:
    api_qa = [
        {"q": "In veloxdb, what is the exact positional argument order of VaultEngine.write_batch?", "a": "records, table_name", "patterns": [r"records.*table_name|records\s*,\s*table"]},
        {"q": "In veloxdb v2.4, which method replaced the deprecated VaultEngine.insert_rows()?", "a": "write_batch", "patterns": [r"\bwrite_batch\b"]},
        {"q": "In veloxdb, what method must be called on a CursorFrame to get a plain list of row dicts?", "a": "materialize", "patterns": [r"\bmaterialize\b"]},
        {"q": "In veloxdb, what is the default operation op of veloxdb.fold_metric?", "a": "sum_sq", "patterns": [r"\bsum_sq\b"]},
        {"q": "In veloxdb, what must be called on VaultEngine immediately before veloxdb.compact_segments(engine)?", "a": "unmount", "patterns": [r"\bunmount\b"]},
        {"q": "In veloxdb, what is the maximum number of records allowed per write_batch call (MAX_BATCH)?", "a": "256", "patterns": [r"\b256\b"]},
        {"q": "In veloxdb, what keyword argument replaced flush_immediately in write_batch?", "a": "sync_wal", "patterns": [r"\bsync_wal\b"]},
        {"q": "In veloxdb, what exception is raised if write_batch is called before mount()?", "a": "StateViolationError", "patterns": [r"StateViolationError"]},
    ]

    code_tasks = [
        {
            "id": "velox_ingest_rows",
            "fn_name": "ingest_rows",
            "prompt": (
                "Using the `veloxdb` library, write a Python function `ingest_rows(db_path: str, table: str, rows: list[dict]) -> int` "
                "that creates a read-write `VaultEngine` at `db_path`, prepares it for I/O, inserts `rows` into `table` with WAL syncing enabled, "
                "closes/unmounts the engine cleanly, and returns the number of inserted rows."
            ),
            "test_code": (
                "import veloxdb\n"
                "veloxdb._STORE.clear()\n"
                "n = ingest_rows('db1', 'metrics', [{'v': 3}, {'v': 4}])\n"
                "assert n == 2, f'Expected 2, got {n}'\n"
                "eng = veloxdb.VaultEngine('db1', mode='ro').mount()\n"
                "got = eng.scan('metrics').materialize()\n"
                "eng.unmount()\n"
                "assert got == [{'v': 3}, {'v': 4}], f'Unexpected rows: {got}'\n"
                "# Ensure engine was unmounted by compacting\n"
                "s = veloxdb.compact_segments(veloxdb.VaultEngine('db1'))\n"
                "assert s['rows_total'] == 2\n"
            ),
            "oracle_code": (
                "import veloxdb\n"
                "def ingest_rows(db_path: str, table: str, rows: list[dict]) -> int:\n"
                "    eng = veloxdb.VaultEngine(db_path, mode='rw')\n"
                "    eng.mount()\n"
                "    try:\n"
                "        count = eng.write_batch(rows, table, sync_wal=True)\n"
                "    finally:\n"
                "        eng.unmount()\n"
                "    return count\n"
            ),
        },
        {
            "id": "velox_query_and_fold",
            "fn_name": "query_and_fold",
            "prompt": (
                "Using the `veloxdb` library, write a Python function "
                "`query_and_fold(db_path: str, table: str, field: str, limit: int = 20) -> tuple[list[dict], float]` "
                "that mounts `VaultEngine` in read-only mode `'ro'`, scans up to `limit` rows from `table`, computes the default "
                "`veloxdb.fold_metric` on `field`, extracts the plain list of dicts, unmounts cleanly, and returns `(rows_list, metric_value)`."
            ),
            "test_code": (
                "import veloxdb\n"
                "veloxdb._STORE.clear()\n"
                "e = veloxdb.VaultEngine('db2').mount()\n"
                "e.write_batch([{'x': 3}, {'x': 4}, {'x': 5}], 'pts')\n"
                "e.unmount()\n"
                "rows, val = query_and_fold('db2', 'pts', 'x', limit=2)\n"
                "assert isinstance(rows, list) and len(rows) == 2 and isinstance(rows[0], dict)\n"
                "assert rows == [{'x': 3}, {'x': 4}]\n"
                "assert abs(val - 25.0) < 1e-6, f'Expected sum_sq=25.0, got {val}'\n"
            ),
            "oracle_code": (
                "import veloxdb\n"
                "def query_and_fold(db_path: str, table: str, field: str, limit: int = 20):\n"
                "    eng = veloxdb.VaultEngine(db_path, mode='ro')\n"
                "    eng.mount()\n"
                "    try:\n"
                "        cur = eng.scan(table, limit=limit)\n"
                "        metric = veloxdb.fold_metric(cur, field)\n"
                "        rows = cur.materialize()\n"
                "    finally:\n"
                "        eng.unmount()\n"
                "    return rows, metric\n"
            ),
        },
        {
            "id": "velox_batch_and_compact",
            "fn_name": "batch_upsert_and_compact",
            "prompt": (
                "Using `veloxdb`, write `batch_upsert_and_compact(db_path: str, table: str, rows: list[dict], floor: float = 0.4) -> dict` "
                "that opens `VaultEngine(db_path, page_kb=32)`, mounts it, writes `rows` into `table` with `sync_wal=False`, "
                "unmounts before compacting with `ratio_floor=floor`, and returns the compaction summary dict."
            ),
            "test_code": (
                "import veloxdb\n"
                "veloxdb._STORE.clear()\n"
                "res = batch_upsert_and_compact('db3', 'logs', [{'a': 1}, {'a': 2}, {'a': 3}], floor=0.45)\n"
                "assert isinstance(res, dict)\n"
                "assert res['page_kb'] == 32\n"
                "assert abs(res['ratio_floor'] - 0.45) < 1e-6\n"
                "assert res['rows_total'] == 3\n"
            ),
            "oracle_code": (
                "import veloxdb\n"
                "def batch_upsert_and_compact(db_path: str, table: str, rows: list[dict], floor: float = 0.4) -> dict:\n"
                "    eng = veloxdb.VaultEngine(db_path, page_kb=32)\n"
                "    eng.mount()\n"
                "    eng.write_batch(rows, table, sync_wal=False)\n"
                "    eng.unmount()\n"
                "    return veloxdb.compact_segments(eng, ratio_floor=floor)\n"
            ),
        },
        {
            "id": "velox_filtered_scan",
            "fn_name": "safe_filtered_scan",
            "prompt": (
                "Using `veloxdb`, write `safe_filtered_scan(db_path: str, table: str, predicate: dict) -> list[dict]` "
                "that opens `VaultEngine(db_path, mode='ro')`, mounts it, scans `table` filtered by `where=predicate`, "
                "returns the materialized list of row dicts, and guarantees unmounting in a `finally` block."
            ),
            "test_code": (
                "import veloxdb\n"
                "veloxdb._STORE.clear()\n"
                "e = veloxdb.VaultEngine('db4').mount()\n"
                "e.write_batch([{'k': 'a', 'v': 1}, {'k': 'b', 'v': 2}, {'k': 'a', 'v': 3}], 'events')\n"
                "e.unmount()\n"
                "out = safe_filtered_scan('db4', 'events', {'k': 'a'})\n"
                "assert out == [{'k': 'a', 'v': 1}, {'k': 'a', 'v': 3}], f'Got {out}'\n"
            ),
            "oracle_code": (
                "import veloxdb\n"
                "def safe_filtered_scan(db_path: str, table: str, predicate: dict) -> list[dict]:\n"
                "    eng = veloxdb.VaultEngine(db_path, mode='ro')\n"
                "    eng.mount()\n"
                "    try:\n"
                "        return eng.scan(table, where=predicate).materialize()\n"
                "    finally:\n"
                "        eng.unmount()\n"
            ),
        },
        {
            "id": "velox_chunked_bulk",
            "fn_name": "chunked_bulk_load",
            "prompt": (
                "Using `veloxdb`, write `chunked_bulk_load(db_path: str, table: str, rows: list[dict]) -> int` "
                "that mounts a read-write `VaultEngine` at `db_path`, inserts `rows` safely in chunks obeying `veloxdb.MAX_BATCH`, "
                "unmounts, and returns the number of `write_batch` calls made."
            ),
            "test_code": (
                "import veloxdb\n"
                "veloxdb._STORE.clear()\n"
                "rows = [{'idx': i} for i in range(600)]\n"
                "batches = chunked_bulk_load('db5', 'bulk', rows)\n"
                "assert batches == 3, f'Expected 3 batches, got {batches}'\n"
                "e = veloxdb.VaultEngine('db5', mode='ro').mount()\n"
                "all_rows = e.scan('bulk', limit=1000).materialize()\n"
                "e.unmount()\n"
                "assert len(all_rows) == 600 and all_rows[-1]['idx'] == 599\n"
            ),
            "oracle_code": (
                "import veloxdb\n"
                "def chunked_bulk_load(db_path: str, table: str, rows: list[dict]) -> int:\n"
                "    eng = veloxdb.VaultEngine(db_path, mode='rw')\n"
                "    eng.mount()\n"
                "    calls = 0\n"
                "    try:\n"
                "        for s in range(0, len(rows), veloxdb.MAX_BATCH):\n"
                "            eng.write_batch(rows[s:s + veloxdb.MAX_BATCH], table)\n"
                "            calls += 1\n"
                "    finally:\n"
                "        eng.unmount()\n"
                "    return calls\n"
            ),
        },
        {
            "id": "velox_migrate",
            "fn_name": "migrate_table",
            "prompt": (
                "Using `veloxdb`, write `migrate_table(db_path: str, src_table: str, dst_table: str) -> int` "
                "that opens a read-write `VaultEngine`, mounts it, scans all rows from `src_table` (using `limit=1000`), "
                "writes the materialized rows into `dst_table` using the v2.4 batch API, unmounts cleanly, and returns the migrated count."
            ),
            "test_code": (
                "import veloxdb\n"
                "veloxdb._STORE.clear()\n"
                "e = veloxdb.VaultEngine('db6').mount()\n"
                "e.write_batch([{'id': 10}, {'id': 20}], 'old_t')\n"
                "e.unmount()\n"
                "cnt = migrate_table('db6', 'old_t', 'new_t')\n"
                "assert cnt == 2\n"
                "e = veloxdb.VaultEngine('db6', mode='ro').mount()\n"
                "assert e.scan('new_t').materialize() == [{'id': 10}, {'id': 20}]\n"
                "e.unmount()\n"
            ),
            "oracle_code": (
                "import veloxdb\n"
                "def migrate_table(db_path: str, src_table: str, dst_table: str) -> int:\n"
                "    eng = veloxdb.VaultEngine(db_path, mode='rw')\n"
                "    eng.mount()\n"
                "    try:\n"
                "        rows = eng.scan(src_table, limit=1000).materialize()\n"
                "        if rows:\n"
                "            eng.write_batch(rows, dst_table)\n"
                "    finally:\n"
                "        eng.unmount()\n"
                "    return len(rows)\n"
            ),
        },
    ]

    rl_train_tasks = [
        {
            "id": "velox_rl_1",
            "fn_name": "mean_metric_for_tag",
            "prompt": (
                "Using `veloxdb`, write `mean_metric_for_tag(db_path: str, table: str, tag: str, field: str) -> float` "
                "that mounts `VaultEngine(db_path, mode='ro')`, scans `table` with `where={'tag': tag}` and `limit=500`, "
                "computes `veloxdb.fold_metric(cursor, field, op='mean')`, unmounts, and returns the float."
            ),
            "test_code": (
                "import veloxdb\n"
                "veloxdb._STORE.clear()\n"
                "e = veloxdb.VaultEngine('rl1').mount()\n"
                "e.write_batch([{'tag': 'a', 'v': 10}, {'tag': 'b', 'v': 99}, {'tag': 'a', 'v': 20}], 't')\n"
                "e.unmount()\n"
                "assert abs(mean_metric_for_tag('rl1', 't', 'a', 'v') - 15.0) < 1e-6\n"
            ),
        },
        {
            "id": "velox_rl_2",
            "fn_name": "write_and_compact_deferred",
            "prompt": (
                "Using `veloxdb`, write `write_and_compact_deferred(db_path: str, table: str, rows: list[dict]) -> int` "
                "that opens `VaultEngine(db_path)`, mounts it, writes `rows` with `sync_wal=False`, unmounts, runs `compact_segments`, "
                "and returns `rows_total` from the summary dict."
            ),
            "test_code": (
                "import veloxdb\n"
                "veloxdb._STORE.clear()\n"
                "assert write_and_compact_deferred('rl2', 'ev', [{'x': 1}, {'x': 2}]) == 2\n"
            ),
        },
        {
            "id": "velox_rl_3",
            "fn_name": "sum_metric_top_n",
            "prompt": (
                "Using `veloxdb`, write `sum_metric_top_n(db_path: str, table: str, field: str, n: int) -> float` "
                "that opens `VaultEngine(db_path, mode='ro')`, mounts, scans `table` with `limit=n`, computes `veloxdb.fold_metric(cur, field, op='sum')`, "
                "unmounts, and returns the sum."
            ),
            "test_code": (
                "import veloxdb\n"
                "veloxdb._STORE.clear()\n"
                "e = veloxdb.VaultEngine('rl3').mount()\n"
                "e.write_batch([{'val': 2}, {'val': 5}, {'val': 8}], 'nums')\n"
                "e.unmount()\n"
                "assert abs(sum_metric_top_n('rl3', 'nums', 'val', 2) - 7.0) < 1e-6\n"
            ),
        },
        {
            "id": "velox_rl_4",
            "fn_name": "append_two_tables",
            "prompt": (
                "Using `veloxdb`, write `append_two_tables(db_path: str, t1: str, r1: list[dict], t2: str, r2: list[dict]) -> int` "
                "that opens `VaultEngine(db_path, mode='rw')`, mounts, writes `r1` to `t1` and `r2` to `t2`, unmounts, and returns total rows written."
            ),
            "test_code": (
                "import veloxdb\n"
                "veloxdb._STORE.clear()\n"
                "assert append_two_tables('rl4', 'a', [{'x': 1}], 'b', [{'x': 2}, {'x': 3}]) == 3\n"
            ),
        },
        {
            "id": "velox_rl_5",
            "fn_name": "read_all_rows",
            "prompt": (
                "Using `veloxdb`, write `read_all_rows(db_path: str, table: str) -> list[dict]` "
                "that mounts `VaultEngine(db_path, mode='ro')`, scans `table` with `limit=1000`, materializes rows, unmounts, and returns them."
            ),
            "test_code": (
                "import veloxdb\n"
                "veloxdb._STORE.clear()\n"
                "e = veloxdb.VaultEngine('rl5').mount()\n"
                "e.write_batch([{'k': 7}], 'q')\n"
                "e.unmount()\n"
                "assert read_all_rows('rl5', 'q') == [{'k': 7}]\n"
            ),
        },
        {
            "id": "velox_rl_6",
            "fn_name": "energy_and_compact",
            "prompt": (
                "Using `veloxdb`, write `energy_and_compact(db_path: str, table: str, rows: list[dict], field: str) -> float` "
                "that mounts `VaultEngine(db_path)`, writes `rows` into `table`, scans `table` with `limit=len(rows)`, "
                "computes default `fold_metric(cur, field)`, unmounts, compacts segments, and returns the float metric."
            ),
            "test_code": (
                "import veloxdb\n"
                "veloxdb._STORE.clear()\n"
                "v = energy_and_compact('rl6', 'sig', [{'p': 3}, {'p': 4}], 'p')\n"
                "assert abs(v - 25.0) < 1e-6\n"
            ),
        },
    ]

    pref_train = [
        {
            "prompt": "Write a veloxdb snippet to insert `rows` into table `'sensors'` on `eng` (already mounted).",
            "chosen": "```python\neng.write_batch(rows, 'sensors', sync_wal=True)\n```",
            "rejected": "```python\neng.insert_rows('sensors', rows, flush_immediately=True)\n```",
        },
        {
            "prompt": "How should you iterate row dicts returned by `cur = eng.scan('logs')` in veloxdb?",
            "chosen": "Call `cur.materialize()` to obtain a plain list of dictionaries, because iterating `cur` directly yields `(row_id, dict)` tuples.",
            "rejected": "Iterate over `cur` directly (`for row in cur:`) because `scan()` returns a plain list of row dictionaries.",
        },
        {
            "prompt": "In veloxdb, how do you compute sum-of-squares on field `'amp'` from `cur = eng.scan('wave')`?",
            "chosen": "Pass the `CursorFrame` directly to `veloxdb.fold_metric(cur, 'amp')` without calling `.materialize()` first.",
            "rejected": "Materialize first with `rows = cur.materialize()` and then call `veloxdb.fold_metric(rows, 'amp', op='sum_sq')`.",
        },
        {
            "prompt": "What order must `unmount()` and `compact_segments()` follow in veloxdb?",
            "chosen": "Always call `eng.unmount()` before calling `veloxdb.compact_segments(eng)`, because compacting a mounted engine raises `StateViolationError`.",
            "rejected": "Call `veloxdb.compact_segments(eng)` while `eng` is still mounted, then call `eng.unmount()` afterward.",
        },
        {
            "prompt": "Write a veloxdb call to write `batch` into `'orders'` without forcing an immediate WAL sync.",
            "chosen": "```python\neng.write_batch(batch, 'orders', sync_wal=False)\n```",
            "rejected": "```python\neng.write_batch('orders', batch, flush_immediately=False)\n```",
        },
        {
            "prompt": "What happens in veloxdb if you pass 300 records in a single `write_batch` call?",
            "chosen": "It raises `BatchLimitError` because `MAX_BATCH` is `256`, so you must chunk the records into slices of at most 256.",
            "rejected": "It automatically splits the batch internally up to 1024 rows without any limit error.",
        },
        {
            "prompt": "How do you open a read-only veloxdb engine and safely scan `'audit'`?",
            "chosen": "```python\neng = veloxdb.VaultEngine(path, mode='ro').mount()\ntry:\n    return eng.scan('audit').materialize()\nfinally:\n    eng.unmount()\n```",
            "rejected": "```python\neng = veloxdb.VaultEngine(path, mode='read')\nreturn eng.scan('audit')\n```",
        },
        {
            "prompt": "What port constant and default page size does veloxdb define?",
            "chosen": "`VELOX_PORT = 7419` and default `page_kb = 16`.",
            "rejected": "`VELOX_PORT = 5432` and default `page_kb = 4`.",
        },
    ]

    pref_eval = [
        {
            "prompt": "Write the correct veloxdb v2.4 line to append `items` to `'events'` on mounted engine `eng`.",
            "chosen": "```python\neng.write_batch(items, 'events')\n```",
            "rejected": "```python\neng.insert_rows('events', items)\n```",
        },
        {
            "prompt": "In veloxdb, how do you get both materialized rows and `fold_metric` on `'score'` from `eng.scan('games')`?",
            "chosen": "```python\ncur = eng.scan('games')\nscore_sq = veloxdb.fold_metric(cur, 'score')\nrows = cur.materialize()\n```",
            "rejected": "```python\nrows = eng.scan('games')\nscore_sq = veloxdb.fold_metric(rows, 'score')\n```",
        },
        {
            "prompt": "Write the clean veloxdb teardown and compaction sequence on mounted `eng` with floor 0.25.",
            "chosen": "```python\neng.unmount()\nveloxdb.compact_segments(eng, ratio_floor=0.25)\n```",
            "rejected": "```python\nveloxdb.compact_segments(eng, ratio_floor=0.25)\neng.unmount()\n```",
        },
        {
            "prompt": "How should you pass `records` and `table` to `VaultEngine.write_batch` with deferred WAL sync?",
            "chosen": "Pass `records` first, `table` second, and keyword-only `sync_wal=False`: `eng.write_batch(records, table, sync_wal=False)`.",
            "rejected": "Pass `table` first, `records` second, and `flush_immediately=False`.",
        },
        {
            "prompt": "What exception does `VaultEngine.insert_rows` raise in veloxdb v2.4?",
            "chosen": "`DeprecationError`, because `insert_rows` was removed in favor of `write_batch(records, table_name)`.",
            "rejected": "`AttributeError` or it silently forwards to `write_batch` for backwards compatibility.",
        },
        {
            "prompt": "Why does `for r in eng.scan('tbl'): r['id']` fail in veloxdb?",
            "chosen": "Because `scan()` returns a `CursorFrame` whose direct iteration yields `(row_id, row_dict)` tuples; you must call `.materialize()` to iterate dicts directly.",
            "rejected": "Because `scan()` returns JSON strings instead of dicts.",
        },
    ]

    return {
        "seed": 42,
        "framework": "veloxdb",
        "mock_module_name": "veloxdb",
        "mock_module_source": VELOXDB_MOCK_SRC,
        "doc_text": VELOXDB_DOC,
        "api_qa": api_qa,
        "code_tasks": code_tasks,
        "neg_controls": _standard_python_neg_controls(),
        "pref_train": pref_train,
        "pref_eval": pref_eval,
        "rl_train_tasks": rl_train_tasks,
    }


def _nexflow_bundle() -> Dict[str, object]:
    api_qa = [
        {"q": "In nexflow, what is the exact positional argument order of PulseGraph.dispatch?", "a": "payload, lane_id", "patterns": [r"payload.*lane_id|payload\s*,\s*lane"]},
        {"q": "In nexflow v3.1, which method replaced the deprecated PulseGraph.send()?", "a": "dispatch", "patterns": [r"\bdispatch\b"]},
        {"q": "In nexflow, what method must be called on a StageTicket before graph.arm() when strict=True?", "a": "seal", "patterns": [r"\bseal\b"]},
        {"q": "In nexflow, what method on a DrainBundle extracts a plain list of values?", "a": "as_values", "patterns": [r"\bas_values\b"]},
        {"q": "In nexflow, what is the positional argument order of nexflow.bind_stage?", "a": "graph, stage_fn, stage_tag", "patterns": [r"graph.*stage_fn.*stage_tag"]},
        {"q": "In nexflow, what is the default priority value in bind_stage?", "a": "10", "patterns": [r"\b10\b"]},
        {"q": "In nexflow, what keyword argument replaced async_mode in dispatch?", "a": "buffered", "patterns": [r"\bbuffered\b"]},
        {"q": "In nexflow, what string constant is stored in NEXFLOW_MAGIC?", "a": "NFX-409", "patterns": [r"NFX-409"]},
    ]

    code_tasks = [
        {
            "id": "nex_single_stage",
            "fn_name": "run_pipeline",
            "prompt": (
                "Using `nexflow`, write `run_pipeline(items: list[int], lane: str) -> list[int]` that creates `PulseGraph('pipe')`, "
                "binds a stage `(lambda x: x * 2)` tagged `'double'`, properly seals and arms the graph, dispatches every item into `lane`, "
                "drains `lane`, and returns the plain list of output values."
            ),
            "test_code": (
                "out = run_pipeline([1, 2, 3], 'main')\n"
                "assert out == [2, 4, 6], f'Expected [2, 4, 6], got {out}'\n"
            ),
            "oracle_code": (
                "import nexflow\n"
                "def run_pipeline(items: list[int], lane: str) -> list[int]:\n"
                "    g = nexflow.PulseGraph('pipe')\n"
                "    t = nexflow.bind_stage(g, lambda x: x * 2, 'double')\n"
                "    t.seal()\n"
                "    g.arm()\n"
                "    for x in items:\n"
                "        g.dispatch(x, lane)\n"
                "    return g.drain(lane).as_values()\n"
            ),
        },
        {
            "id": "nex_priority_order",
            "fn_name": "ordered_two_stage",
            "prompt": (
                "Using `nexflow`, write `ordered_two_stage(items: list[int]) -> list[int]` that builds a `PulseGraph('ord', workers=4)`, "
                "binds stage `'add_one'` (`lambda x: x + 1`) with `priority=20` and stage `'mul_ten'` (`lambda x: x * 10`) with `priority=5`, "
                "seals both tickets, arms the graph, dispatches `items` onto lane `'out'` with `buffered=True`, and returns the drained values."
            ),
            "test_code": (
                "out = ordered_two_stage([2, 3])\n"
                "assert out == [21, 31], f'Expected [21, 31], got {out}'\n"
            ),
            "oracle_code": (
                "import nexflow\n"
                "def ordered_two_stage(items: list[int]) -> list[int]:\n"
                "    g = nexflow.PulseGraph('ord', workers=4)\n"
                "    nexflow.bind_stage(g, lambda x: x + 1, 'add_one', priority=20).seal()\n"
                "    nexflow.bind_stage(g, lambda x: x * 10, 'mul_ten', priority=5).seal()\n"
                "    g.arm()\n"
                "    for x in items:\n"
                "        g.dispatch(x, 'out', buffered=True)\n"
                "    return g.drain('out').as_values()\n"
            ),
        },
        {
            "id": "nex_reverse_drain",
            "fn_name": "drain_reversed_pairs",
            "prompt": (
                "Using `nexflow`, write `drain_reversed_pairs(items: list[str], lane: str) -> list[tuple[int, str]]` "
                "that creates `PulseGraph('rev')`, binds `str.upper` tagged `'up'`, seals, arms, dispatches `items` into `lane`, "
                "drains `lane` with `keep_order=False`, and returns the list of `(seq_no, value)` tuples directly from the `DrainBundle`."
            ),
            "test_code": (
                "out = drain_reversed_pairs(['a', 'b', 'c'], 'fast')\n"
                "assert out == [(2, 'C'), (1, 'B'), (0, 'A')], f'Got {out}'\n"
            ),
            "oracle_code": (
                "import nexflow\n"
                "def drain_reversed_pairs(items: list[str], lane: str):\n"
                "    g = nexflow.PulseGraph('rev')\n"
                "    nexflow.bind_stage(g, str.upper, 'up').seal()\n"
                "    g.arm()\n"
                "    for it in items:\n"
                "        g.dispatch(it, lane)\n"
                "    return list(g.drain(lane, keep_order=False))\n"
            ),
        },
        {
            "id": "nex_rearm_append",
            "fn_name": "reconfigure_and_run",
            "prompt": (
                "Using `nexflow`, write `reconfigure_and_run(first: list[int], second: list[int]) -> list[int]` "
                "that binds `'inc'` (`lambda x: x + 1`), seals, arms, dispatches `first` to `'q'`, disarms the graph, "
                "binds `'sq'` (`lambda x: x * x`, `priority=20`), seals it, re-arms, dispatches `second` to `'q'`, and returns drained values."
            ),
            "test_code": (
                "out = reconfigure_and_run([1], [2, 3])\n"
                "assert out == [2, 9, 16], f'Expected [2, 9, 16], got {out}'\n"
            ),
            "oracle_code": (
                "import nexflow\n"
                "def reconfigure_and_run(first: list[int], second: list[int]) -> list[int]:\n"
                "    g = nexflow.PulseGraph('reconf')\n"
                "    nexflow.bind_stage(g, lambda x: x + 1, 'inc', priority=10).seal()\n"
                "    g.arm()\n"
                "    for x in first:\n"
                "        g.dispatch(x, 'q')\n"
                "    g.disarm()\n"
                "    nexflow.bind_stage(g, lambda x: x * x, 'sq', priority=20).seal()\n"
                "    g.arm()\n"
                "    for x in second:\n"
                "        g.dispatch(x, 'q')\n"
                "    return g.drain('q').as_values()\n"
            ),
        },
        {
            "id": "nex_multi_lane",
            "fn_name": "split_lanes",
            "prompt": (
                "Using `nexflow`, write `split_lanes(nums: list[int]) -> tuple[list[int], list[int]]` that builds a `PulseGraph('split')`, "
                "binds `(lambda x: x + 5)` tagged `'shift'`, seals and arms, dispatches even numbers to lane `'even'` and odd numbers to `'odd'`, "
                "and returns `(even_values, odd_values)`."
            ),
            "test_code": (
                "ev, od = split_lanes([1, 2, 3, 4])\n"
                "assert ev == [7, 9] and od == [6, 8], f'Got {(ev, od)}'\n"
            ),
            "oracle_code": (
                "import nexflow\n"
                "def split_lanes(nums: list[int]):\n"
                "    g = nexflow.PulseGraph('split')\n"
                "    nexflow.bind_stage(g, lambda x: x + 5, 'shift').seal()\n"
                "    g.arm()\n"
                "    for n in nums:\n"
                "        g.dispatch(n, 'even' if n % 2 == 0 else 'odd')\n"
                "    return g.drain('even').as_values(), g.drain('odd').as_values()\n"
            ),
        },
        {
            "id": "nex_tag_magic",
            "fn_name": "tag_with_magic",
            "prompt": (
                "Using `nexflow`, write `tag_with_magic(msgs: list[str], lane: str) -> list[str]` that builds a `PulseGraph('tag')`, "
                "binds a stage formatting each message as `f\"{nexflow.NEXFLOW_MAGIC}:{msg}\"` with tag `'header'`, seals, arms, "
                "dispatches `msgs` to `lane`, and returns drained string values."
            ),
            "test_code": (
                "out = tag_with_magic(['ping', 'pong'], 'ctrl')\n"
                "assert out == ['NFX-409:ping', 'NFX-409:pong'], f'Got {out}'\n"
            ),
            "oracle_code": (
                "import nexflow\n"
                "def tag_with_magic(msgs: list[str], lane: str) -> list[str]:\n"
                "    g = nexflow.PulseGraph('tag')\n"
                "    nexflow.bind_stage(g, lambda m: f'{nexflow.NEXFLOW_MAGIC}:{m}', 'header').seal()\n"
                "    g.arm()\n"
                "    for m in msgs:\n"
                "        g.dispatch(m, lane)\n"
                "    return g.drain(lane).as_values()\n"
            ),
        },
    ]

    rl_train_tasks = [
        {
            "id": "nex_rl_1",
            "fn_name": "scale_and_drain",
            "prompt": "Using `nexflow`, write `scale_and_drain(xs: list[int], factor: int) -> list[int]` using `PulseGraph`, `bind_stage`, `.seal()`, `.arm()`, `dispatch`, and `drain('s').as_values()`.",
            "test_code": "assert scale_and_drain([2, 5], 3) == [6, 15]\n",
        },
        {
            "id": "nex_rl_2",
            "fn_name": "strip_and_lower",
            "prompt": "Using `nexflow`, write `strip_and_lower(texts: list[str]) -> list[str]` with two sealed stages: `'strip'` (`priority=1`) and `'low'` (`priority=2`), returning `drain('txt').as_values()`.",
            "test_code": "assert strip_and_lower([' HI ', ' Bye']) == ['hi', 'bye']\n",
        },
        {
            "id": "nex_rl_3",
            "fn_name": "dispatch_seq_ids",
            "prompt": "Using `nexflow`, write `dispatch_seq_ids(n: int) -> list[int]` that arms an empty `PulseGraph('seq')`, dispatches `i` to `'ids'` `n` times, and returns the list of returned `seq_no` integers.",
            "test_code": "assert dispatch_seq_ids(4) == [0, 1, 2, 3]\n",
        },
        {
            "id": "nex_rl_4",
            "fn_name": "buffered_lane_sum",
            "prompt": "Using `nexflow`, write `buffered_lane_sum(xs: list[int]) -> int` that binds `(lambda x: x + 1)`, seals, arms, dispatches with `buffered=True` to `'b'`, and returns `sum(g.drain('b').as_values())`.",
            "test_code": "assert buffered_lane_sum([1, 2, 3]) == 9\n",
        },
        {
            "id": "nex_rl_5",
            "fn_name": "max_retry_prefix",
            "prompt": "Using `nexflow`, write `max_retry_prefix(items: list[int]) -> list[int]` that adds `nexflow.MAX_RETRIES` to each item via a sealed stage and returns drained values from `'r'`.",
            "test_code": "assert max_retry_prefix([1, 4]) == [4, 7]\n",
        },
        {
            "id": "nex_rl_6",
            "fn_name": "reverse_values",
            "prompt": "Using `nexflow`, write `reverse_values(xs: list[int]) -> list[int]` that arms `PulseGraph('rv')`, dispatches `xs` to `'x'`, and drains with `keep_order=False` returning `.as_values()`.",
            "test_code": "assert reverse_values([10, 20, 30]) == [30, 20, 10]\n",
        },
    ]

    pref_train = [
        {
            "prompt": "Write a nexflow line to dispatch `event` onto lane `'telemetry'` on armed graph `g`.",
            "chosen": "```python\ng.dispatch(event, 'telemetry')\n```",
            "rejected": "```python\ng.send('telemetry', event)\n```",
        },
        {
            "prompt": "How do you attach and prepare a stage function `fn` tagged `'parse'` on `PulseGraph` `g`?",
            "chosen": "```python\nnexflow.bind_stage(g, fn, 'parse').seal()\n```",
            "rejected": "```python\nnexflow.bind_stage(g, 'parse', fn)\n```",
        },
        {
            "prompt": "How do you extract output values from `bundle = g.drain('main')` in nexflow?",
            "chosen": "Call `bundle.as_values()`, because iterating `bundle` directly yields `(seq_no, value)` tuples.",
            "rejected": "Return `list(bundle)` directly, because `drain()` yields plain values.",
        },
        {
            "prompt": "What keyword argument controls buffering in `PulseGraph.dispatch` in nexflow v3.1?",
            "chosen": "Keyword-only `buffered=False` (or `buffered=True`); `async_mode=` raises `LegacyRouteError`.",
            "rejected": "`async_mode=True`.",
        },
        {
            "prompt": "Why does `g.arm()` raise `UnarmedGraphError` on `PulseGraph('x', strict=True)`?",
            "chosen": "Because one or more bound `StageTicket` objects were not sealed via `.seal()` prior to calling `g.arm()`.",
            "rejected": "Because `strict=True` forbids having more than one stage.",
        },
        {
            "prompt": "In what order do two nexflow stages with `priority=20` and `priority=5` execute?",
            "chosen": "Ascending priority order: the stage with `priority=5` executes first, followed by `priority=20`.",
            "rejected": "Descending priority order: `priority=20` runs before `priority=5`.",
        },
        {
            "prompt": "Can `nexflow.bind_stage` be called while `graph` is already armed?",
            "chosen": "No; it raises `UnarmedGraphError`. You must call `graph.disarm()` first, bind and seal the new stage, and re-arm.",
            "rejected": "Yes, `bind_stage` attaches hot stages dynamically on an armed graph.",
        },
        {
            "prompt": "What is `nexflow.NEXFLOW_MAGIC` and `nexflow.MAX_RETRIES`?",
            "chosen": "`NEXFLOW_MAGIC = 'NFX-409'` and `MAX_RETRIES = 3`.",
            "rejected": "`NEXFLOW_MAGIC = 'NX-100'` and `MAX_RETRIES = 5`.",
        },
    ]

    pref_eval = [
        {
            "prompt": "Write the correct nexflow v3.1 call to push `msg` to `'alerts'` with buffering enabled.",
            "chosen": "```python\ng.dispatch(msg, 'alerts', buffered=True)\n```",
            "rejected": "```python\ng.send('alerts', msg, async_mode=True)\n```",
        },
        {
            "prompt": "Write the clean nexflow snippet to register `fn` as `'clean'` at priority 3, seal, and arm `g`.",
            "chosen": "```python\nnexflow.bind_stage(g, fn, 'clean', priority=3).seal()\ng.arm()\n```",
            "rejected": "```python\nnexflow.bind_stage(g, 'clean', fn, priority=3)\ng.arm()\n```",
        },
        {
            "prompt": "How do you obtain a plain list of values from `g.drain('jobs')` in nexflow?",
            "chosen": "```python\ng.drain('jobs').as_values()\n```",
            "rejected": "```python\ng.drain('jobs')\n```",
        },
        {
            "prompt": "What exception does `PulseGraph.send()` raise in nexflow v3.1?",
            "chosen": "`LegacyRouteError`, because `send()` was replaced by `dispatch(payload, lane_id)`.",
            "rejected": "It aliases directly to `dispatch` without error.",
        },
        {
            "prompt": "What must you call before `bind_stage()` if `g` is already armed?",
            "chosen": "`g.disarm()`, then bind and `.seal()` the stage before calling `g.arm()` again.",
            "rejected": "Nothing; `bind_stage` automatically pauses the graph.",
        },
        {
            "prompt": "Why does `[x + 1 for x in g.drain('nums')]` fail in nexflow?",
            "chosen": "Because `DrainBundle` yields `(seq_no, value)` tuples on iteration; use `g.drain('nums').as_values()` first.",
            "rejected": "Because `DrainBundle` yields string representations of integers.",
        },
    ]

    return {
        "seed": 1337,
        "framework": "nexflow",
        "mock_module_name": "nexflow",
        "mock_module_source": NEXFLOW_MOCK_SRC,
        "doc_text": NEXFLOW_DOC,
        "api_qa": api_qa,
        "code_tasks": code_tasks,
        "neg_controls": _standard_python_neg_controls(),
        "pref_train": pref_train,
        "pref_eval": pref_eval,
        "rl_train_tasks": rl_train_tasks,
    }


def _standard_python_neg_controls() -> List[Dict[str, str]]:
    """Standard Python collections / numpy / itertools negative-control executable tasks."""
    return [
        {
            "id": "std_counter_top",
            "fn_name": "top_k_chars",
            "prompt": "Using Python's `collections.Counter`, write `top_k_chars(text: str, k: int) -> list[tuple[str, int]]` returning the `k` most common non-space characters in `text`.",
            "test_code": "res = top_k_chars('abracadabra a', 2)\nassert res[0] == ('a', 6) and res[1][0] in ('b', 'r')\n",
        },
        {
            "id": "std_numpy_zscore",
            "fn_name": "zscore_array",
            "prompt": "Using `numpy`, write `zscore_array(vals: list[float]) -> list[float]` that returns the zero-mean unit-std normalized values as a Python list of floats (if std == 0, return zeros).",
            "test_code": "import numpy as np\nout = zscore_array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])\nassert abs(sum(out)) < 1e-6 and abs(np.std(out) - 1.0) < 1e-6\nassert zscore_array([3.0, 3.0]) == [0.0, 0.0]\n",
        },
        {
            "id": "std_defaultdict_group",
            "fn_name": "group_by_initial",
            "prompt": "Using `collections.defaultdict`, write `group_by_initial(words: list[str]) -> dict[str, list[str]]` grouping non-empty words by lowercase first letter while preserving order.",
            "test_code": "out = group_by_initial(['Apple', 'ant', 'Boat', ''])\nassert out == {'a': ['Apple', 'ant'], 'b': ['Boat']}\n",
        },
        {
            "id": "std_deque_rotate",
            "fn_name": "sliding_window_max",
            "prompt": "Using Python (e.g. `collections.deque` or slicing), write `sliding_window_max(nums: list[int], k: int) -> list[int]` returning the maximum of every contiguous window of length `k`.",
            "test_code": "assert sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]\n",
        },
        {
            "id": "std_numpy_dot",
            "fn_name": "cosine_sim",
            "prompt": "Using `numpy`, write `cosine_sim(a: list[float], b: list[float]) -> float` returning the cosine similarity between vectors `a` and `b` (0.0 if either norm is 0).",
            "test_code": "assert abs(cosine_sim([1.0, 0.0], [0.0, 1.0])) < 1e-6\nassert abs(cosine_sim([3.0, 4.0], [6.0, 8.0]) - 1.0) < 1e-6\n",
        },
        {
            "id": "std_itertools_pairs",
            "fn_name": "pairwise_diffs",
            "prompt": "Write a Python function `pairwise_diffs(nums: list[int]) -> list[int]` returning `[b - a for a, b in zip(nums, nums[1:])]`.",
            "test_code": "assert pairwise_diffs([5, 9, 3, 10]) == [4, -6, 7] and pairwise_diffs([1]) == []\n",
        },
    ]
