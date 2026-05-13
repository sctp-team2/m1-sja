You are building a Streamlit app for interactive analysis of large datasets.

ARCHITECTURE CONSTRAINTS
- Streamlit is the UI layer. DuckDB does heavy computation WHEN the user opts in.
- Data is stored as pickle (pkl) files (or a pickle (pkl) directory partitioned by date/region).
- The user explicitly chooses the execution mode via a sidebar toggle:
  (a) "Load into DuckDB" (checked)   → use the optimised DuckDB path below.
  (b) "Load into DuckDB" (unchecked) → load directly into a pandas DataFrame via pd.read_pickle (pkl) / pd.read_csv and do all filtering and aggregation in pandas.
- The selected mode must be stored in st.session_state and drive which query function is called. Both paths must produce DataFrames with identical schema so downstream display code does not branch.
- Show a small caption under the toggle indicating which mode is active and a one-line trade-off hint (e.g. "DuckDB mode: scales to large files. Pandas mode: simpler, holds full dataset in memory.").

EXECUTION MODES

Mode A — DuckDB path (when toggle is ON):
- pandas is used ONLY to hold small, already-aggregated result sets returned from DuckDB for display and charting.
- Never load the full dataset into a pandas DataFrame.
- All filtering, grouping, joins, and aggregations happen inside DuckDB SQL.
- Read pickle (pkl) directly: `SELECT ... FROM 'data/*.pickle (pkl)' WHERE ...` — let DuckDB push down predicates and column pruning.
- For repeated access, register a view once at connection setup: `CREATE VIEW sales AS SELECT * FROM 'data/sales/*.pickle (pkl)'`.
- Return results with `.df()` for pandas (or `.pl()` for Polars).
- Use parameterised SQL via DuckDB's `?` placeholders or named parameters. Never f-string user input into SQL.

Mode B — Pandas-only path (when toggle is OFF):
- Load the file once into a pandas DataFrame and cache the loaded DataFrame with `@st.cache_data` keyed on the file path and mtime.
- Apply filters via boolean indexing / `.query()` and aggregations via `groupby().agg()`.
- Warn the user (via `st.warning`) if the file size on disk exceeds a configurable threshold (default 500 MB) and recommend switching to DuckDB mode.
- Do not attempt to chunk or stream in this mode — keep the implementation simple and obvious; the whole point of this path is "fits in memory, easy to reason about."

CACHING REQUIREMENTS (mandatory, applies to both modes)
1. DuckDB connection: wrap in `@st.cache_resource` so a single connection is reused across sessions and reruns. Only created on first use of Mode A.
   - Function returns `duckdb.connect(database=":memory:", read_only=False)` and registers any pickle (pkl) views.
2. Query results: wrap every query function in `@st.cache_data(ttl=...)` with an explicit TTL appropriate to data freshness (e.g. 3600s for daily data, 300s for near-real-time).
   - Cache key must include the execution mode AND all filter parameters as function arguments (dates, regions, categories, etc.) so different mode/filter combos cache independently.
   - Return small DataFrames only (post-aggregation). Never cache raw scans of large tables in DuckDB mode.
3. For unhashable inputs (e.g. DuckDB connection), use the `_` underscore prefix on the parameter name to skip hashing: `def run_query_duckdb(_con, start_date, end_date): ...`
4. Use `@st.cache_data(show_spinner="Querying...")` so the user sees feedback.
5. Expose a "Clear cache" button in the sidebar using `st.cache_data.clear()`. Clearing cache should NOT reset the user's mode toggle.
6. Switching the toggle does not by itself invalidate the cache — because mode is part of the cache key, each path retains its own cached results.

DISPATCH PATTERN
- Implement a single public function `get_results(filters: dict) -> pd.DataFrame` that internally inspects `st.session_state["use_duckdb"]` and dispatches to either `_query_duckdb(...)` or `_query_pandas(...)`.
- Both internal functions must return DataFrames with the same columns, dtypes, and sort order. Add a lightweight assertion in development to catch schema drift.

UI PATTERNS
- Sidebar holds: (1) the "Load into DuckDB" checkbox at the top, (2) filters (date range, dropdowns, multiselects), (3) a "Clear cache" button.
- Each filter is a function argument to the cached query function.
- Main panel shows: (a) headline metrics via `st.metric`, (b) one or two charts via `st.altair_chart` or `st.plotly_chart`, (c) a paginated `st.dataframe` of the result set.
- Display the active execution mode as a small badge near the page title (e.g. `st.caption("Engine: DuckDB")` or `st.caption("Engine: pandas")`) so it is obvious which path served the current view.
- Use `st.session_state` for user-driven state (selected row, mode toggle), not for caching data.
- Wrap chart-building in its own `@st.cache_data` function keyed on the DataFrame's hash if construction is expensive.

ANTI-PATTERNS TO REJECT
- Branching display code on mode — display layer must be mode-agnostic.
- Loading the full pickle (pkl)/CSV into pandas in DuckDB mode.
- Creating a new DuckDB connection per query.
- Caching DataFrames larger than ~50 MB.
- Storing query results in `st.session_state` instead of using `@st.cache_data`.
- String-concatenated SQL.
- Silently switching modes — the user controls the toggle.

DELIVERABLE
- A single `app.py` with: connection factory, two query functions (`_query_duckdb`, `_query_pandas`), a `get_results` dispatcher, sidebar mode toggle + filter widgets, main display, and a sidebar cache-clear button.
- A short `README.md` explaining how to run it, where to drop data files, and when to use each mode.
- A `requirements.txt` with pinned versions of streamlit, duckdb, pandas, and (optionally) polars and altair.

Ask clarifying questions about: dataset schema, expected row counts, partition strategy, freshness requirements, and the size threshold at which the pandas-mode warning should fire — before writing code.