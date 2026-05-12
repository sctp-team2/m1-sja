# MCF Recruiter Insights Dashboard — Build Specification

## Document purpose

This is a build spec for a Streamlit dashboard for HR analysts and recruiters analyzing MyCareersFuture (MCF) Singapore job posting data. Hand this document to Claude (or any developer) to scaffold and implement the app. All product decisions, design choices, and acceptance criteria are captured here.

The data layer is already complete — see `feature_engineering.py` in the same repository. Do not re-implement feature engineering; consume the output frame.

---

## 1. Business context

### Objective
Identify promising job categories and roles by comparing vacancies, average salary, experience requirements, views, and application counts.

### Target users
HR analysts and recruiters at end-employers and recruitment agencies in Singapore.

### Value proposition
Help recruiters:
1. Identify high-demand roles and categories worth prioritizing.
2. Benchmark salary offers against market peer groups (within-category, within-seniority).
3. Prioritize recruitment strategies for roles with many vacancies but low application rates.

The "high vacancies, low applications" intersection is the **primary headline insight** the app must surface — see Page 3 Section 3.2.

---

## 2. Tech stack and project layout

### Core dependencies
- **Python** 3.10+ (required — `feature_engineering.py` uses `X | None` type hints)
- **Streamlit** 1.40+ (for native multi-page apps)
- **Pandas** 2.x
- **Plotly Express** + **Plotly Graph Objects** for interactive charts
- **PyArrow** for Parquet I/O

### File structure
```
sctp-dsai/
├── feature_engineering.py          # existing, do not modify
├── recruiter_dashboard_spec.md     # this file
├── data/
│   └── mcf_features.parquet        # pre-built feature dataframe (~80 MB)
├── app/
│   ├── app.py                      # main entry, landing page
│   ├── pages/
│   │   ├── 1_📊_Overview.py
│   │   ├── 2_🔍_Deep_Analysis.py
│   │   └── 3_📋_Recruitment_Report.py
│   └── lib/
│       ├── __init__.py
│       ├── data_loader.py          # cached load + filter helpers
│       ├── filters.py              # shared sidebar filter widget
│       ├── chart_helpers.py        # palette, themed plotly defaults
│       └── suggestions.py          # recruiter recommendation engine
└── requirements.txt
```

### Build the data file once (not in app)
Run a one-off script that does:
```python
import pandas as pd
from feature_engineering import build_features

df_raw = pd.read_parquet("path/to/raw_mcf.parquet")  # or however raw data is loaded
df = build_features(df_raw)
df.to_parquet("data/mcf_features.parquet")
```
The app then loads this Parquet on startup. Re-run this script whenever the raw data refreshes.

---

## 3. Data layer

### Available features (from `feature_engineering.build_features`)

The cached frame has 67 columns: 21 original MCF fields plus 46 derived features. The full inventory is documented in `feature_engineering.py`. The app should lean heavily on these derived columns rather than re-computing.

**Salary & seniority benchmarking**
- `salary_band` — categorical: `<3k`, `3-5k`, `5-8k`, `8-12k`, `12-20k`, `20k+`
- `salary_zscore_within_category` — normalized vs category peers
- `salary_percentile_within_category` — [0, 1]
- `salary_premium_vs_category_median` — SGD diff
- `salary_per_yoe` — pay per year of required experience
- `title_seniority` — ordered: `C-suite, Director/Head, Manager/Lead, Senior, Mid/Other, Junior`

**Demand & engagement metrics**
- `applications_per_view`, `applications_per_vacancy`, `views_per_vacancy`
- `demand_intensity_score` — z-scored, clipped ±5
- `views_zscore_within_category`
- `apps_per_view_vs_category_median`

**Hard-to-fill signals**
- `hard_to_fill_score` — composite, z-scored, clipped ±5
- `posting_duration_days`, `metadata_repostCount`, `is_reposted`
- `posting_quality_score`, `hidden_gem_flag`

**Quality flags (use as filters)**
- `is_agency`, `is_reposted`, `mass_hiring_flag`, `zero_engagement_flag`
- `salary_undisclosed_flag`, `salary_suspicious_low`, `seniority_mismatch_flag`

**Time**
- `posting_year`, `posting_month`, `posting_quarter`, `posting_year_month`, `posting_cohort_quarter`
- `days_since_posting`, `is_active_at_snapshot`

**Company**
- `postedCompany_name`, `company_posting_count`, `company_posting_bucket`
- `company_avg_salary`, `company_salary_spread`
- `company_category_diversity`, `company_tenure_days`

### Loading pattern

```python
# app/lib/data_loader.py
import streamlit as st
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parents[2] / "data" / "mcf_features.parquet"

@st.cache_data(show_spinner="Loading MCF feature data...")
def load_features() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)
```

Every page calls `load_features()` then applies filters. The cache is shared across pages, so the parquet only loads once per session.

---

## 4. App architecture

### Multi-page setup
Use Streamlit's native multi-page convention (numbered files in `/pages`). The sidebar persistently shows:
1. Auto-generated page navigation (top).
2. **Shared filters** that apply across all pages.
3. Active filter summary chip.
4. Reset button.

### Shared filter system

Implement `render_sidebar_filters(df)` in `lib/filters.py` that returns a dict of selections. State persists in `st.session_state` so users don't lose context when switching pages.

**Filter inventory (sidebar, top to bottom):**

1. **Date range** — `st.date_input` on `metadata_originalPostingDate`. Default: full data range.
2. **Category** — `st.multiselect` on `category_1`. Default: empty (means "all").
3. **Seniority** — `st.multiselect` on `title_seniority`. Default: empty.
4. **Salary band** — `st.multiselect` on `salary_band`. Default: empty.
5. **Experience** — `st.slider` on `minimumYearsExperience` (0–20). Default: 0–20.
6. **Employment type** — `st.multiselect` on `employmentTypes`. Default: empty.
7. **Quality filters** (inside `st.expander("Data quality", expanded=False)`):
   - Exclude `zero_engagement_flag` — checked by default
   - Exclude `mass_hiring_flag` — checked by default
   - Exclude `salary_suspicious_low` — unchecked
   - Employer type — radio: `Both / Direct only / Agency only` (default Both)

The applied filter mask should be computed once per page render and reused.

```python
# Conceptual shape — implement in lib/data_loader.py
def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if filters["categories"]:
        mask &= df["category_1"].isin(filters["categories"])
    if filters["exclude_zero_engagement"]:
        mask &= ~df["zero_engagement_flag"]
    # ... etc
    return df[mask]
```

---

## 5. Page 1 — Overview

### Goal
Five-second orientation: what does the market look like right now, who are the big players, where's the pay.

### Layout (top to bottom)

**Header**
- Title: "Singapore Job Market Overview"
- Subtitle: dynamic — "Showing {n:,} postings from {min_date} to {max_date}"

**KPI strip (5 cards, `st.columns(5)`)**
Each card uses `st.metric()` with delta vs unfiltered baseline (small grey "vs. all data" comparator).
- Total postings (filtered count)
- Median average salary (S$X,XXX)
- Top category (single string)
- Median posting duration (days)
- Share of roles reposted (%)

**Two-column section (`st.columns(2)`)**
- **Left**: Top 15 categories by posting count — horizontal bar, sorted descending. X-axis: count. Color: single brand blue.
- **Right**: Top 15 companies by posting count — horizontal bar. Color: split by `is_agency` (agency vs direct, two-color legend).

**Salary section**
- Histogram of `average_salary` (50 bins, clipped at S$25k for readability). Overlay vertical lines at median and p75. Hover shows count + salary range + which `salary_band` bin.
- Below: Box plot of `average_salary` by `positionLevels`, sorted by median ascending. Annotate sample size on each box.

**Employment & seniority mix (`st.columns(2)`)**
- Donut: `employmentTypes` distribution.
- Donut: `title_seniority` distribution.

### Interactivity
- All charts respond to sidebar filters.
- Standard Plotly tooltips: exact counts, percentages, and (where applicable) the underlying salary band.
- No special click-to-filter behavior required for v1 (it adds state complexity for marginal value).

### How this serves the business objective
Orients users to the market landscape before they drill in. Establishes baseline expectations for what "normal" looks like in each segment.

---

## 6. Page 2 — Deep Analysis

### Goal
Surface the *non-obvious* signals: which roles are hard to fill, where demand outstrips supply, which postings are underperforming. Every section is framed as **Problem → Solution → Chart → Insight → Action** so the recruiter sees not just data but interpretation.

### Section format
Every section on this page follows the same pattern. Implement this as a reusable function:

```python
def render_analysis_section(title, problem, solution, chart_fn, insight_fn, action_fn, df):
    st.subheader(title)
    with st.container(border=True):
        col_text, col_chart = st.columns([1, 2])
        with col_text:
            st.markdown(f"**The problem.** {problem}")
            st.markdown(f"**Our approach.** {solution}")
        with col_chart:
            chart_fn(df)
        st.markdown(f"**Key insight.** {insight_fn(df)}")
        st.markdown(f"**Recommended action.** {action_fn(df)}")
```

The Insight and Action strings should be **generated from the filtered df**, not hardcoded — they should change as the user filters.

### Sections

**2.1 Where is demand outstripping supply?**

- *Problem*: Recruiters need to know which segments have lots of applicants per opening (competitive) vs few applicants per opening (hard to attract).
- *Solution*: `demand_intensity_score` aggregated by `category_1` × `title_seniority`.
- *Chart*: Heatmap, diverging `RdBu_r` colormap (red = high competition, blue = hard to attract). Annotate cells with median score.
- *Insight*: "The most competitive segment is {top_cell}: {N}× the median applicants per vacancy."
- *Action*: "Consider raising hiring bar in competitive cells; broaden requirements in low-competition ones."

**2.2 Which roles are hardest to fill?**

- *Problem*: Some postings stay open for months and need repeated reposting. Surface them early.
- *Solution*: `hard_to_fill_score` = composite of `posting_duration_days`, `metadata_repostCount`, and inverse `applications_per_vacancy`, z-scored and clipped at ±5.
- *Chart*: Top 20 leaderboard table with `st.dataframe`. Columns: title, category, company, duration (days), reposts, apps/vacancy, hard_to_fill_score. Use `column_config.ProgressColumn` for the score.
- *Note in UI*: Mass-hiring rows are excluded by default in the sidebar — toggle off if you want them.
- *Insight*: "Top {N} hard-to-fill roles cluster in {top_category}. Mean duration {M} days, average {R} reposts."
- *Action*: "For these roles, consider direct sourcing, broader YoE requirements, or compensation review."

**2.3 Salary positioning**

- *Problem*: "Is my S$X offer competitive for a {category} {seniority} role?"
- *Solution*: `salary_zscore_within_category` and `salary_percentile_within_category`.
- *Chart*: Violin/box plot of `average_salary` by `category_1`, with markers at p25, median, p75. Hover shows count and z-score reference.
- *Below chart — salary calculator widget*: three inputs (category dropdown, seniority dropdown, YoE slider). Returns median + p25/p75 + sample size for that segment. Use `st.form` to batch input.
- *Insight*: "Within {category}, the 75th-percentile salary is S${p75}. Roles paying below S${p25} sit in the bottom quartile."
- *Action*: "Benchmark your offers against this distribution before posting."

**2.4 Engagement funnel**

- *Problem*: "My posting got views but few applications — why?"
- *Solution*: `apps_per_view_vs_category_median` flags posts converting below their category norm.
- *Chart*: Scatter `metadata_totalNumberOfView` (x) vs `metadata_totalNumberJobApplication` (y), color by `apps_per_view_vs_category_median` (RdBu_r). Sample to 5,000 points for performance.
- *Insight*: "{N} postings have above-median views but below-median application rates."
- *Action*: "Audit JD clarity, title specificity, or stated requirements for these underperformers."

**2.5 Hidden gems**

- *Problem*: Some postings are quiet but well-targeted — high conversion despite low traffic. Worth studying.
- *Solution*: `hidden_gem_flag` (bottom 25% views, top 25% conversion).
- *Chart*: Sortable table of top 20 hidden gems: title, category, company, views, apps, conversion rate.
- *Insight*: "{N} hidden-gem postings identified. {top_category} dominates."
- *Action*: "Review JD language from these for what works in niche markets."

**2.6 Agency vs direct employer**

- *Problem*: Recruitment agencies post differently from end-employers. Aggregate stats can mislead.
- *Solution*: `is_agency` flag (detected from company-name keywords).
- *Chart*: Faceted comparison — four small panels in a 2×2 grid: salary distribution, conversion rate, posting duration, role mix by seniority. Each panel splits by `is_agency`.
- *Insight*: "Agencies post {N}% of roles. Median salary difference: S${diff}."
- *Action*: "Filter to direct employers when benchmarking salaries for comparable roles."

---

## 7. Page 3 — Recruitment Report

### Goal
A scrollable, exportable executive summary that directly answers the business objective: *"What are the most promising job categories and roles to prioritize?"* This is the page recruiters return to weekly to plan their pipeline.

### Section 3.1 — Executive summary strip

Four cards (`st.columns(4)`) with hero metrics:
1. **Most promising category** — the `category_1` with the highest count of promising roles (defined in 3.2).
2. **Best benchmark target** — category with cleanest salary distribution (largest n, smallest undisclosed share).
3. **Highest-demand seniority tier** — seniority with highest median `demand_intensity_score`.
4. **Largest market segment** — top `category_1` by total postings.

Each card shows the headline value plus one-line context.

### Section 3.2 — Promising roles (the headline)

Operational definition of "promising":
```
numberOfVacancies >= 3
AND applications_per_vacancy <= 25th percentile of its category
AND NOT mass_hiring_flag
AND NOT zero_engagement_flag
AND NOT salary_suspicious_low
```

Output: Top 30 table sorted by `numberOfVacancies` descending.

Columns:
- title, category_1, postedCompany_name, numberOfVacancies, applications_per_vacancy, average_salary, days_since_posting, hard_to_fill_score.

Use `st.dataframe` with sortable columns and `column_config.NumberColumn(format="S$%d")` for salary.

Below the table: `st.download_button` to export as CSV.

### Section 3.3 — Category-level recommendations

For each of the top 10 categories by posting count, render a card with:
- Category name (header)
- Median salary + p25–p75 range
- Demand intensity rank (1 = highest demand)
- Median time-to-fill (`posting_duration_days`)
- **Strategic recommendation** (auto-generated):
  - **High-priority**: many vacancies AND low apps/vacancy AND short median duration → "Focus sourcing here; the market is open."
  - **Steady state**: metrics near the overall median → "Maintain current strategy."
  - **Saturated**: high apps/vacancy AND below-median salary → "Plenty of supply; selectivity is key."

Render as a 2-column grid of `st.container(border=True)` cards.

### Section 3.4 — Salary benchmark report

Cross-tab table: `category_1` × `title_seniority` → median salary, p25, p75, n.

Filterable by `minimumYearsExperience` band (slider above the table). Cells where n < 10 should be greyed out or marked "—" (insufficient sample).

Below: download button for the salary matrix as CSV.

### Section 3.5 — Recruiter action list

Run the suggestion engine (`lib/suggestions.py`) across the filtered df. Aggregate counts by suggestion type:

- "X roles need compensation review" (salary in bottom 20% of category)
- "Y roles flagged as hard-to-fill"
- "Z roles with low conversion despite high views"
- "W roles with title/seniority mismatch"
- "V reposted roles open >60 days"

Each line is an `st.expander` that reveals the actual rows when clicked.

### Section 3.6 — Export

- Download full filtered dataset (CSV)
- Download promising-roles table (CSV)
- Download salary benchmark matrix (CSV)

PDF export is a stretch goal; skip for v1.

### How this serves the business objective
This page is the operational deliverable. The Promising Roles table (3.2), Category Recommendations (3.3), and Action List (3.5) directly translate the data into a recruiting plan.

---

## 8. Suggestion engine — `lib/suggestions.py`

### Public API

```python
def suggest_for_row(row: pd.Series) -> list[dict]:
    """Return list of {severity, type, message} for a single posting."""

def suggest_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply suggest_for_row across df. Returns long-form:
    columns = [row_index, severity, type, message]."""
```

### Rule set

| Trigger | Severity | Type | Message |
|---|---|---|---|
| `salary_percentile_within_category < 0.2` | High | compensation | "Pay is in bottom 20% for {category}. Median in this category is S${median}." |
| `hard_to_fill_score > 2.0` | Medium | sourcing | "Above-market difficulty to fill. Consider direct sourcing or broader requirements." |
| `posting_quality_score < -1.5` AND NOT `zero_engagement_flag` | Medium | listing | "Posting attracts views but underperforms on conversion. Review JD specificity." |
| `seniority_mismatch_flag` | Low | categorization | "Title seniority ({title_seniority}) doesn't match position level ({positionLevels}). Verify intended level." |
| `applications_per_vacancy > p75 of category` | Low | shortlisting | "High applicant volume; tighten shortlist criteria." |
| `is_reposted` AND `posting_duration_days > 60` | Medium | strategy | "Reposted and open >60 days. Reassess JD or compensation." |
| `salary_undisclosed_flag` | Low | transparency | "Salary not disclosed. Postings with ranges typically get higher application rates." |

### Display

Use color-coded markdown headers or `st.alert` boxes:
- High → 🔴 red
- Medium → 🟡 amber
- Low → 🔵 blue

Each suggestion **must reference the feature that triggered it** so recruiters trust the recommendation.

---

## 9. Design choices

### Color palette (semantic, accessible)

| Use | Color | Hex |
|---|---|---|
| Primary brand | Deep blue | `#1E3A8A` |
| Secondary | Slate | `#64748B` |
| Success / positive | Green | `#10B981` |
| Warning | Amber | `#F59E0B` |
| Critical | Red | `#DC2626` |
| Background | Off-white | `#F8FAFC` |
| Text primary | Near-black | `#0F172A` |
| Text muted | Grey | `#64748B` |

For diverging scales (z-scores, premiums): use Plotly `RdBu_r` so red = high, blue = low.
For sequential scales (counts, salaries): use `Blues` or `Viridis`.

Centralize these in `lib/chart_helpers.py`:

```python
PALETTE = {
    "primary": "#1E3A8A",
    "secondary": "#64748B",
    # ...
}

def themed_layout(fig, title=None):
    """Apply consistent typography and spacing to a Plotly figure."""
    fig.update_layout(
        font_family="Inter, system-ui, sans-serif",
        title=dict(text=title, font=dict(size=16)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig
```

### Typography
- Sans-serif throughout. Plotly default sans-serif is fine.
- 12pt chart body, 16pt chart titles.
- No serif. No color-as-only-information (always include label or shape).

### Layout
- `st.set_page_config(layout="wide", page_title="MCF Recruiter Insights")` on every page.
- Use `st.columns` for KPI strips and side-by-side charts.
- Use `st.container(border=True)` for grouped content (especially Page 2 sections).
- Use `st.expander` to manage information density on Page 3.

### Readability
- Limit each page to ≤7 main visual elements.
- Every chart has a title and a one-line caption explaining what it shows.
- Use `st.dataframe` for tabular data — better than HTML tables.
- Format numbers ≥1,000 with comma separators; prefix salaries with "S$".

---

## 10. Implementation notes

### Performance
- Cache the feature load (`@st.cache_data` on `load_features()`). One-time cost.
- Filter operations on 1M rows take <100ms via pandas boolean mask — no further optimization needed.
- For repeated category-level aggregates (Pages 2 and 3), cache them too:

```python
@st.cache_data
def category_aggregates(df_signature: tuple, df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("category_1", observed=True).agg(...)
```

Pass a stable hashable signature (e.g., `(len(df), df.index.min(), df.index.max())`) as first arg to control cache invalidation when filters change.

### Edge cases the app must handle
- **Empty filtered df**: every chart should render "No data matches current filters" with a reset hint.
- **Single-row categories**: z-score charts skip these with a footnote.
- **Null `applications_per_view`**: treat as 0 for chart aggregations, but NaN in tables.
- **Plotly scatter > 50k points**: sample down to ~5,000 with a notice.

### Error handling
Wrap each page's main render in a try/except that displays a friendly error message if anything fails, with the option to reset filters.

---

## 11. Acceptance criteria

The app is "done" when:

1. All three pages load without errors on the full dataset.
2. Sidebar filters propagate across all pages and persist via `st.session_state`.
3. Page 2 sections all follow the Problem/Solution/Chart/Insight/Action structure with dynamic insight/action text.
4. Page 3 Section 3.2 ("Promising Roles") correctly implements the defined criteria.
5. The suggestion engine produces grounded, feature-traceable recommendations.
6. CSV exports work on the three download buttons in Page 3.
7. Color palette and typography are consistent across pages.
8. Page load is <3 seconds after the parquet is cached.
9. The app gracefully handles empty filter results.

---

## 12. Out of scope for v1 (future enhancements)

Document these in a TODO file but **do not build** in the first version:

- **Skill tag extraction** — parse `title` for tech keywords (Python, Java, AWS, SAP, .NET, React, etc.) and add as a multi-select filter and as a chart dimension. Implementation hint: build a regex-based extractor in a new `add_skill_tags(df)` function in `feature_engineering.py`.
- **Geographic features** — extract Singapore districts from titles (Clementi, Jurong, CBD, Tampines, etc.) and add a map view.
- **PDF report export** — generate a styled PDF of Page 3 using ReportLab or WeasyPrint.
- **Saved filter presets** — let recruiters bookmark common filter combinations (e.g., "IT mid-senior $5–8k").
- **Comparison mode** — side-by-side analysis of two filter slices (e.g., "Banking vs IT for senior roles").
- **Dedicated trend page** — time-series view of postings, salaries, seasonality.
- **Authentication / user accounts** — if shared across an organization.

---

## 13. Build instructions for the implementing developer

1. Create the file structure exactly as specified in section 2.
2. Generate the Parquet data file by running `feature_engineering.build_features()` on the raw MCF data once. Save to `data/mcf_features.parquet`.
3. Build pages incrementally: Page 1 first (verify filters and caching work end-to-end), then Page 2, then Page 3.
4. Use Plotly Express where possible; drop to `go.Figure` only for custom annotations.
5. For every dynamic Insight/Action string on Page 2, derive the values from the filtered df — never hardcode.
6. Test edge cases explicitly: empty filter result, single-category filter, max-zoom date range, filters that exclude every row.
7. Add a brief docstring at the top of each page file explaining the page's purpose, the business question it answers, and the features it consumes.
8. Do not modify `feature_engineering.py`. If a needed transformation is missing, add a TODO note rather than monkey-patching.
9. When ambiguity arises, default to: simplicity over cleverness, prose explanations over jargon, and direct support for the business objective.

---

## 14. Reference — the business question on every screen

Every page should make it possible to answer some part of this composite question:

> *"For the segment I care about, which roles are most worth prioritizing — high vacancy count, manageable competition, fair salary expectations — and what should I do differently to fill them?"*

If a chart, table, or widget doesn't help answer some piece of this, it doesn't belong in the app.