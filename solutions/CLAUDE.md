# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 1. Project summary

MCF Singapore job-posting analytics for HR analysts and recruiters. The feature pipeline is complete and validated; the Streamlit recruiter dashboard is fully specified but **not yet built**. v1 scope is the dashboard only — see `recruiter-dashboard.md`.

## 2. Current state

- **Done:** `feature_engineering.py` — 10 pipeline blocks, 46 derived columns on top of 21 raw, smoke-test passes.
- **Specified, not built:** 3-page Streamlit app (Overview, Deep Analysis, Recruitment Report) per `recruiter-dashboard.md`.
- **Out of scope for v1:** skill-tag extraction, geographic features, PDF export, auth, live data refresh / scraping.

## 3. Key files

| Path | Role |
| --- | --- |
| `feature_engineering.py` | Pipeline — do NOT modify casually; re-run validation if you do. |
| `recruiter-dashboard.md` | Full app spec — follow exactly. Section 12 lists v2 items. |
| `data/mcf_features.pkl` | Built feature frame (gitignored, regenerate as needed). |

The sibling `5m-data-*` / `6m-data-*` folders in the wider workspace are unrelated SCTP coursework. Ignore them.

## 4. Conventions

- **Pure functions.** Every `add_X_features(df, ...)` returns a new frame. No in-place mutation. Blocks are independently re-runnable.
- **Memory-efficient dtypes.** `int8` / `int16` for counts/ages, `float32` for derived numerics, `bool` for flags, `category` for low-cardinality strings, `datetime64[ns]` for dates.
- **Categorical groupby.** Always pass `observed=True` to avoid cartesian-product blowups.
- **Block order matters in `build_features()`.** Later blocks depend on earlier outputs (e.g. composite scores need `bench_df`). Read the `build_features` docstring before reordering.
- **Naming.** `_flag` (bool), `_score` (z-scored numeric), `_band`/`_bucket` (categorical), `_within_category` (peer-normalized), `_vs_category_median` (raw diff).

## 5. Data quirks (read before analysing)

1. **MCF salary ceiling ~S$20k/month.** Top values cluster at S$19,500–19,999.50. Senior/exec roles are right-censored — upper-tail benchmarks are unreliable.
2. **`salary_undisclosed_flag` is 0% on this dataset.** The flag is defensive in case the upstream extract changes.
3. **Mass-hiring rows (~1.5%, `numberOfVacancies > 10`) distort raw stats.** Exclude them from leaderboards by default — that's what `bench_df` is for.
4. **Agency postings are ~21% of the dataset** (keyword detection: RECRUIT, STAFFING, MANPOWER, etc.). Separate them via `is_agency` for clean salary benchmarks.
5. **Index has gaps.** 1,012,928 rows but max index ~1,048,574 — do not assume a contiguous integer index.
6. **Composite scores are z-scored and clipped to ±5.** `hard_to_fill_score`, `demand_intensity_score`, `posting_quality_score`.

## 6. Common commands

```bash
# Smoke-test pipeline (runs synthetic data, prints 46 derived columns)
python feature_engineering.py

# Rebuild the feature pickle — one-liner that loads raw, runs
# build_features, and writes data/mcf_features.pkl
python -c "from feature_engineering import build_features; import pandas as pd; \
  build_features(pd.read_pickle('../data/clean_job_step1.pkl')).to_pickle('data/mcf_features.pkl')"

# Run the app (once built)
streamlit run app/app.py
```

Python 3.10+ required (the pipeline uses `X | None` type hints).

## 7. What NOT to do

- Don't modify `feature_engineering.py` without re-running validation (shape == 67, NaN audit, within-category z-score mean≈0/std≈1).
- Don't build features that `recruiter-dashboard.md` section 12 lists as v2 (skill tags, geographic features, PDF export) without confirming scope.
- Don't pull the broader SCTP coursework (5m-/6m-data folders) into this dashboard project.
- Don't drop rows from the main df to "clean" it — keep noisy/dead/outlier postings visible to recruiters. The `bench_df` pattern in `clean_for_benchmarks` is the clean-subset mechanism.

---

For deeper context on the app build, read `recruiter-dashboard.md` (app spec, acceptance criteria, scope boundaries).
