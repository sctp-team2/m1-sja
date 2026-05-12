# MCF Recruiter Insights — Report

Singapore job-market analytics for HR analysts and recruiters, built on
1,012,928 MyCareersFuture (MCF) job postings from October 2022 to May
2024.

---

## 1. Business question

> *"For the segment I care about, which roles are most worth
> prioritizing — high vacancy count, manageable competition, fair salary
> expectations — and what should I do differently to fill them?"*

**Target users.** HR analysts and recruiters at end-employers and
recruitment agencies in Singapore.

**Value proposition.** Help recruiters do three things:

1. Identify high-demand roles and categories worth prioritizing.
2. Benchmark salary offers against market peer groups
   (within-category, within-seniority).
3. Spot roles with many vacancies but low application rates so they
   can be sourced differently.

The **"high vacancies, low applications"** intersection is the headline
insight the dashboard must surface — translating raw posting data into
a recruiting plan.

---

## 2. Data process

### Source
The raw dataset is `clean_job_step1.pkl` — a 1,012,928-row × 21-column
pandas frame produced upstream from the MCF CSV dump. Each row is one
job posting with employment type, posting/expiry dates, repost count,
application count, view count, vacancies, position level, company,
salary min/max/average, status, title, category fields, and a hybrid
flag.

### Cleaning (step 1, prior to this submission)
Type-cast each column to a memory-efficient dtype
(`int8`/`int16`/`float32`/`category`/`bool`/`datetime64[ns]`), enforce
schema, and resolve null handling. Pickled the result for reuse.

### Feature engineering (step 3, `feature_engineering.py`)
A single 10-block pipeline (`build_features`) turns the 21 raw columns
into a 67-column feature frame:

| Block | Adds |
| --- | --- |
| 1. Time / lifecycle | `posting_duration_days`, `repost_lag_days`, posting year/month/quarter/day-of-week, `is_reposted` |
| 2. Salary | `salary_range`, `salary_range_pct`, `salary_band` (6 buckets), `salary_per_yoe` |
| 3. Engagement | `applications_per_view`, `applications_per_vacancy`, `views_per_vacancy`, `low_engagement_flag` |
| 4. Title | `title_seniority` (ordered: C-suite → Junior, regex-driven), word/char counts |
| 5. Company | `company_posting_count`, `company_posting_bucket`, `company_avg_salary`, `is_agency` |
| 6. Quality flags | `salary_undisclosed_flag`, `salary_suspicious_low`, `mass_hiring_flag`, `zero_engagement_flag`, `seniority_mismatch_flag` |
| 7. Normalization | `salary_zscore_within_category`, `salary_percentile_within_category`, `salary_premium_vs_category_median`, `views_zscore_within_category`, `apps_per_view_vs_category_median` |
| 8. Recency | `days_since_posting`, `posting_cohort_quarter`, `is_active_at_snapshot` |
| 9. Company enrichments | `company_category_diversity`, `company_salary_spread`, `company_tenure_days` |
| 10. Composite scores | `hard_to_fill_score`, `demand_intensity_score`, `posting_quality_score`, `hidden_gem_flag` (all z-scored, clipped ±5) |

Every block is a pure function (no in-place mutation); the orchestrator
runs them in a fixed dependency order. The full pipeline takes ~5
seconds on 1M rows. The output is pickled to `data/mcf_features.pkl`
(~236 MB) — the single source the dashboard reads.

The notebook `m1-feature-engineering-step3.ipynb` walks through each
block on a stratified sample and reproduces the validation checks
(shape == 67, NaN audit, within-category z-score sanity).

### Output → dashboard
The Streamlit app caches the feature pickle once per session via
`@st.cache_data` and applies sidebar filters through a hashable
filter-signature key, so changing a filter re-fires aggregations once
and serves them from cache thereafter.

---

## 3. Dashboard

A three-page Streamlit app at `app/app.py`. Every page shares a sidebar
with persistent filters (`st.session_state`): date range, category,
seniority, salary band, years of experience, employment type, and a
data-quality expander (exclude zero-engagement / mass-hiring /
suspicious-low salaries; employer-type radio for Direct/Agency/Both).
Every chart on every page has an `ℹ️ How to read this chart` expander.

### Page 1 — Overview
Five-second orientation: 5 KPI cards, top 15 categories and companies,
salary histogram with median + P75 vlines, position-level box plot,
and employment / seniority mix donuts.

### Page 2 — Deep Analysis
Six sections, each framed as **Problem → Solution → Chart → Insight →
Action**. Problem and Solution sit inside a collapsed `💡 Why this
section?` expander so the chart leads.

- **2.1 Demand vs supply** — heatmap of `demand_intensity_score` over
  `category_1 × title_seniority`.
- **2.2 Hardest to fill** — top-20 leaderboard sorted by
  `hard_to_fill_score`.
- **2.3 Salary positioning** — box plot by category + interactive
  segment calculator (category × seniority × YoE → P25 / median / P75).
- **2.4 Engagement funnel** — views × applications scatter, coloured by
  `apps_per_view_vs_category_median`.
- **2.5 Hidden gems** — bottom-25% views, top-25% conversion.
- **2.6 Agency vs direct** — 2×2 facet of salary, conversion, duration,
  and seniority mix.

### Page 3 — Recruitment Report
Operational summary. Executive-summary strip, a top-30 *Promising
Roles* table (the headline output), 10-card category recommendations
with auto-generated guidance (`High-priority` / `Steady state` /
`Saturated`), a category × seniority salary matrix, an actionable
suggestion list grouped by rule severity, and three CSV exports.

### Performance
First page load is 2–3 s on 1M rows. Subsequent renders with the same
filter set are cached — expander clicks recompute nothing. The full
filtered CSV export is encoded only on explicit click (5–10 s) to keep
page loads snappy.

---

## 4. Insights

These are computed against the full dataset with defaults applied
(zero-engagement and mass-hiring postings excluded, both employer
types included).

### 4.1 Three categories dominate the market
The top three categories — **Admin/Secretarial (98,927 postings),
Engineering (98,677), and Information Technology (97,464)** — each
account for almost 10% of the dataset and together cover ~30% of all
postings. Recruiters benchmarking against "the market" are largely
benchmarking against these three.

### 4.2 F&B is where vacancies outnumber applicants
Of 117,980 postings matching the *promising* definition (≥3 vacancies,
applications per vacancy in the bottom quartile of category, excluding
mass-hiring/dead/suspicious rows), **F&B leads with 12,781 promising
postings** — more than IT (10,662), Customer Service (9,199), or
Admin (9,194). The same pattern shows in the hard-to-fill ranking:
**35 of the top 100 hardest-to-fill roles are in F&B**, with Customer
Service (18) and Personal Care / Beauty (9) close behind. This is the
Singapore-specific labour-shortage story the dashboard surfaces.

### 4.3 Median salary is S$3,850/month, capped at S$20k
The salary distribution is concentrated: P25 = S$3,000, median =
S$3,850, P75 = S$5,500. The maximum is **exactly S$20,000** —
MCF imposes a hard ceiling, so executive and senior management roles
are right-censored and any "top-tail" benchmark for these segments is
unreliable. This is flagged in-app on the salary histogram.

### 4.4 Agency vs direct: salary gap is essentially zero
Recruitment agencies post **21.1%** of all roles in the dataset. The
median salary for agency postings (S$3,900) is only **S$100 above**
the median for direct employers (S$3,800). The folk wisdom that
agencies depress pay is not supported in aggregate — though the *mix*
of roles differs (agencies skew toward mid-tier roles), so within-
category comparisons should still split by `is_agency`.

### 4.5 Seniority pay curve is non-monotonic at the top
Median salary climbs cleanly from Junior (S$3,000) through Mid/Other
(S$3,500), Senior (S$5,000), and Manager/Lead (S$6,100) — then
plateaus or inverts at C-suite (S$8,000) and Director/Head (S$10,500).
The C-suite < Director/Head inversion is an artefact of the S$20k
ceiling, which hits C-suite (smaller sample, more extreme roles)
harder than Director/Head. Treat any senior-tier salary report with
this caveat.

### 4.6 Zero-engagement listings are 17% — a meaningful blind spot
**175,127 postings** (17.3%) have no views *and* no applications.
These are dead listings — typically expired but still in the dump.
The dashboard excludes them by default for benchmarking, but they
remain visible if a recruiter wants to inspect what's going stale.

### 4.7 Hidden gems are a small, niche signal
Only **4,913 postings (0.49%)** satisfy `hidden_gem_flag` (bottom 25%
views and top 25% conversion). They cluster in Engineering (907) and
IT (642) — niche, well-targeted roles whose JD language is worth
studying. The signal is small but specific.

---

## Out of scope (v1)

Documented in `recruiter-dashboard.md` §12 as future work:

- Skill-tag extraction from titles (Python, AWS, SAP, …)
- Geographic features (Singapore districts parsed from titles)
- PDF report export
- Saved filter presets, comparison mode
- Live MCF API scraping
- Authentication

---

## Repo & run

See `README.md`. Setup is three steps: `conda activate pds`, build the
feature pickle from `clean_job_step1.pkl`, then `streamlit run
app/app.py`.

Source layout:

```
solutions/
├── feature_engineering.py            10-block pipeline
├── m1-feature-engineering-step3.ipynb feature-engineering walkthrough
├── recruiter-dashboard.md            build spec
├── REPORT.md                         this file
├── README.md
├── data/mcf_features.pkl             built feature frame (gitignored)
└── app/                              Streamlit dashboard (lib + 3 pages)
```
