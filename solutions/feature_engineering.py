"""
Feature engineering for MCF job postings dataset.

Pattern: each `add_*_features` function takes a DataFrame and returns a new one
with extra columns. They're pure (no in-place mutation), so you can re-run any
block while iterating without corrupting your raw data.

Usage in a notebook:
    from feature_engineering import build_features
    df = build_features(df_raw)

Or call individual blocks:
    df = add_time_features(df_raw.copy())
    df = add_salary_features(df)
    ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Time / lifecycle features
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive posting lifecycle and seasonality features."""
    df = df.copy()

    df["posting_duration_days"] = (
        df["metadata_expiryDate"] - df["metadata_newPostingDate"]
    ).dt.days.astype("int16")

    df["repost_lag_days"] = (
        df["metadata_newPostingDate"] - df["metadata_originalPostingDate"]
    ).dt.days.astype("int16")

    posted = df["metadata_originalPostingDate"]
    df["posting_year"] = posted.dt.year.astype("int16")
    df["posting_month"] = posted.dt.month.astype("int8")
    df["posting_quarter"] = posted.dt.quarter.astype("int8")
    df["posting_dow"] = posted.dt.day_name().astype("category")
    # Period -> str -> category keeps it chart-friendly and memory-light
    df["posting_year_month"] = posted.dt.to_period("M").astype(str).astype("category")

    df["is_reposted"] = (df["metadata_repostCount"] > 0)

    return df


# ---------------------------------------------------------------------------
# Salary features
# ---------------------------------------------------------------------------

def add_salary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive salary spread, banding, and pay-per-experience."""
    df = df.copy()

    df["salary_range"] = (df["salary_maximum"] - df["salary_minimum"]).astype("int32")

    # Normalized spread; guard against div-by-zero
    df["salary_range_pct"] = np.where(
        df["average_salary"] > 0,
        df["salary_range"] / df["average_salary"],
        np.nan,
    ).astype("float32")

    # Salary bands tuned for SG market (SGD/month)
    bins = [0, 3000, 5000, 8000, 12000, 20000, np.inf]
    labels = ["<3k", "3-5k", "5-8k", "8-12k", "12-20k", "20k+"]
    df["salary_band"] = pd.cut(
        df["average_salary"], bins=bins, labels=labels, right=False
    )  # already categorical

    # +1 avoids divide-by-zero for 0-YoE roles
    df["salary_per_yoe"] = (
        df["average_salary"] / (df["minimumYearsExperience"] + 1)
    ).astype("float32")

    return df


# ---------------------------------------------------------------------------
# Engagement features
# ---------------------------------------------------------------------------

def add_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive applicant funnel and demand-intensity metrics."""
    df = df.copy()

    df["applications_per_view"] = np.where(
        df["metadata_totalNumberOfView"] > 0,
        df["metadata_totalNumberJobApplication"] / df["metadata_totalNumberOfView"],
        np.nan,
    ).astype("float32")

    df["applications_per_vacancy"] = np.where(
        df["numberOfVacancies"] > 0,
        df["metadata_totalNumberJobApplication"] / df["numberOfVacancies"],
        np.nan,
    ).astype("float32")

    df["views_per_vacancy"] = np.where(
        df["numberOfVacancies"] > 0,
        df["metadata_totalNumberOfView"] / df["numberOfVacancies"],
        np.nan,
    ).astype("float32")

    # "Eyeballs but no clicks": high views, low conversion
    view_p75 = df["metadata_totalNumberOfView"].quantile(0.75)
    apv_p25 = df["applications_per_view"].quantile(0.25)
    df["low_engagement_flag"] = (
        (df["metadata_totalNumberOfView"] >= view_p75)
        & (df["applications_per_view"] <= apv_p25)
    )

    return df


# ---------------------------------------------------------------------------
# Title / content features
# ---------------------------------------------------------------------------

# Order matters: most senior pattern wins via np.select
_SENIORITY_PATTERNS = [
    (r"\b(?:chief|ceo|cfo|cto|coo|cmo|cio|cxo)\b", "C-suite"),
    (r"\b(?:director|vp|vice\s+president|head\s+of)\b", "Director/Head"),
    (r"\b(?:manager|lead|principal)\b", "Manager/Lead"),
    (r"\b(?:senior|sr\.?|snr)\b", "Senior"),
    (r"\b(?:junior|jr\.?|intern|trainee|apprentice|entry[-\s]?level)\b", "Junior"),
]


def add_title_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse seniority signals and structural features from job title."""
    df = df.copy()

    title_lower = df["title"].astype(str).str.lower()

    conditions = [
        title_lower.str.contains(pat, regex=True, na=False)
        for pat, _ in _SENIORITY_PATTERNS
    ]
    values = [label for _, label in _SENIORITY_PATTERNS]

    df["title_seniority"] = pd.Categorical(
        np.select(conditions, values, default="Mid/Other"),
        categories=["C-suite", "Director/Head", "Manager/Lead", "Senior", "Mid/Other", "Junior"],
        ordered=True,
    )

    df["title_word_count"] = df["title"].str.split().str.len().astype("int16")
    df["title_char_length"] = df["title"].str.len().astype("int16")
    df["has_secondary_category"] = df["category_2"].notna()

    return df


# ---------------------------------------------------------------------------
# Company / employer features
# ---------------------------------------------------------------------------

# Common SG recruitment-agency keywords
_AGENCY_PATTERN = (
    r"\b(?:RECRUIT|RECRUITMENT|STAFFING|MANPOWER|SEARCH|TALENT|"
    r"HR\s+SOLUTION|HUMAN\s+RESOURCE|CONSULTING|CONSULTANCY|"
    r"PERSONNEL|PEOPLE\s+SOLUTION|EMPLOYMENT\s+AGENCY)\b"
)


def add_company_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive employer-level rollups and agency flag.

    Company rollups are computed on a small per-company frame, then merged
    back — avoids `transform()` overhead on a high-cardinality category column.
    """
    df = df.copy()

    # Per-company aggregates (observed=True needed because the column is categorical)
    company_stats = (
        df.groupby("postedCompany_name", observed=True)
        .agg(
            company_posting_count=("metadata_jobPostId", "count"),
            company_avg_salary=("average_salary", "mean"),
        )
        .reset_index()
    )
    company_stats["company_posting_count"] = company_stats["company_posting_count"].astype("int32")
    company_stats["company_avg_salary"] = company_stats["company_avg_salary"].astype("float32")

    df = df.merge(company_stats, on="postedCompany_name", how="left")

    # Agency vs. direct employer
    company_upper = df["postedCompany_name"].astype(str).str.upper()
    df["is_agency"] = company_upper.str.contains(_AGENCY_PATTERN, regex=True, na=False)

    # Posting-volume buckets
    bins = [0, 1, 5, 20, 100, np.inf]
    labels = ["1", "2-5", "6-20", "21-100", "100+"]
    df["company_posting_bucket"] = pd.cut(
        df["company_posting_count"], bins=bins, labels=labels, right=True
    )

    return df


# ---------------------------------------------------------------------------
# Data quality flags
# ---------------------------------------------------------------------------

# Coarse mapping from MCF positionLevels -> our title_seniority buckets.
# Anything not in the map produces NaN and is skipped in the mismatch check.
_POSITION_LEVEL_TO_SENIORITY = {
    "Fresh/entry level": "Junior",
    "Non-executive": "Junior",
    "Junior Executive": "Junior",
    "Executive": "Mid/Other",
    "Professional": "Mid/Other",
    "Senior Executive": "Senior",
    "Middle Management": "Manager/Lead",
    "Manager": "Manager/Lead",
    "Senior Manager": "Manager/Lead",
    "Senior Management": "Director/Head",
    "Top Management": "C-suite",
}


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag suspicious, undisclosed, or low-quality postings.

    Recruiters should typically *exclude* these from market benchmarks,
    but they're highly valuable as filter toggles in the UI.

    Depends on: add_title_features (uses title_seniority for mismatch check).
    """
    df = df.copy()

    # Salary undisclosed: min == max is the MCF convention for "not disclosed"
    df["salary_undisclosed_flag"] = (df["salary_minimum"] == df["salary_maximum"])

    # Suspiciously low: below SG realistic floor for most roles (~$1500/mo)
    df["salary_suspicious_low"] = (df["salary_minimum"] < 1500)

    # Bulk hiring (skews per-job aggregates)
    df["mass_hiring_flag"] = (df["numberOfVacancies"] > 10)

    # Dead listing: no views AND no applications
    df["zero_engagement_flag"] = (
        (df["metadata_totalNumberOfView"] == 0)
        & (df["metadata_totalNumberJobApplication"] == 0)
    )

    # Title says one seniority, positionLevels says another
    if "title_seniority" in df.columns:
        pos_seniority = (
            df["positionLevels"].astype(str).map(_POSITION_LEVEL_TO_SENIORITY)
        )
        title_seniority_str = df["title_seniority"].astype(str)
        df["seniority_mismatch_flag"] = (
            pos_seniority.notna()
            & (pos_seniority != title_seniority_str)
            # Don't flag Mid/Other — it's a catch-all, not a real signal
            & (title_seniority_str != "Mid/Other")
        )
    else:
        df["seniority_mismatch_flag"] = False

    return df


# ---------------------------------------------------------------------------
# Within-category normalization
# ---------------------------------------------------------------------------

def add_normalized_features(
    df: pd.DataFrame,
    group_col: str = "category_1",
) -> pd.DataFrame:
    """Compare each row's salary/engagement to peers in the same category.

    This is the magic block for recruiter UX: "this role pays 1.2σ above
    its category median" is far more actionable than a raw salary number.

    Depends on: add_salary_features, add_engagement_features.

    Note: benchmarks are computed on ALL rows. For cleaner benchmarks
    (excluding undisclosed salaries), filter df before calling this and
    keep a separate "clean benchmarks" frame.
    """
    df = df.copy()

    # --- Salary normalization ---
    sal_grp = df.groupby(group_col, observed=True)["average_salary"]
    cat_mean = sal_grp.transform("mean")
    cat_std = sal_grp.transform("std").replace(0, np.nan)
    cat_median = sal_grp.transform("median")

    df["salary_zscore_within_category"] = (
        (df["average_salary"] - cat_mean) / cat_std
    ).astype("float32")

    df["salary_premium_vs_category_median"] = (
        df["average_salary"] - cat_median
    ).astype("float32")

    df["salary_percentile_within_category"] = (
        sal_grp.rank(pct=True).astype("float32")
    )

    # --- Engagement normalization ---
    views_grp = df.groupby(group_col, observed=True)["metadata_totalNumberOfView"]
    views_mean = views_grp.transform("mean")
    views_std = views_grp.transform("std").replace(0, np.nan)
    df["views_zscore_within_category"] = (
        (df["metadata_totalNumberOfView"] - views_mean) / views_std
    ).astype("float32")

    if "applications_per_view" in df.columns:
        apv_median = (
            df.groupby(group_col, observed=True)["applications_per_view"]
            .transform("median")
        )
        df["apps_per_view_vs_category_median"] = (
            df["applications_per_view"] - apv_median
        ).astype("float32")

    return df


# ---------------------------------------------------------------------------
# Recency & cohort features
# ---------------------------------------------------------------------------

def add_recency_features(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Time-since features anchored to a snapshot date.

    Defaults to the most recent posting date in the dataset (historical
    snapshot). Pass a specific date for "as-of" analysis in a Streamlit
    date-picker.
    """
    df = df.copy()

    if snapshot_date is None:
        snapshot_date = df["metadata_originalPostingDate"].max()
    snapshot_date = pd.Timestamp(snapshot_date)

    df["days_since_posting"] = (
        (snapshot_date - df["metadata_originalPostingDate"]).dt.days
    ).astype("int16")

    df["posting_cohort_quarter"] = (
        df["metadata_originalPostingDate"]
        .dt.to_period("Q")
        .astype(str)
        .astype("category")
    )

    # Was the role open at the snapshot date?
    df["is_active_at_snapshot"] = (
        (df["metadata_newPostingDate"] <= snapshot_date)
        & (df["metadata_expiryDate"] >= snapshot_date)
    )

    return df


# ---------------------------------------------------------------------------
# Extended company enrichments
# ---------------------------------------------------------------------------

def add_company_enrichments(df: pd.DataFrame) -> pd.DataFrame:
    """Deeper employer-level signals beyond the basic rollup.

    Lets the Streamlit app answer questions like "what are this company's
    typical pay bands?" or "how diverse is their hiring?"
    """
    df = df.copy()

    extras = (
        df.groupby("postedCompany_name", observed=True)
        .agg(
            company_category_diversity=("category_1", "nunique"),
            company_salary_spread=("average_salary", "std"),
            company_first_seen=("metadata_originalPostingDate", "min"),
            company_last_seen=("metadata_originalPostingDate", "max"),
        )
        .reset_index()
    )
    extras["company_category_diversity"] = extras["company_category_diversity"].astype("int16")
    extras["company_salary_spread"] = extras["company_salary_spread"].astype("float32")

    df = df.merge(extras, on="postedCompany_name", how="left")

    df["company_tenure_days"] = (
        (df["company_last_seen"] - df["company_first_seen"]).dt.days
    ).astype("int16")

    return df


# ---------------------------------------------------------------------------
# Composite business indices
# ---------------------------------------------------------------------------

def _safe_zscore(s: pd.Series) -> pd.Series:
    """Z-score that returns zeros instead of NaN when std is zero/undefined."""
    s = s.astype("float32")
    std = s.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index, dtype="float32")
    return ((s - s.mean()) / std).astype("float32")


def add_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Build interpretable composite indices for recruiter UX.

    All scores are roughly z-distributed (mean 0, std 1), so:
      score >  1.0  → notably above market
      score < -1.0  → notably below market
      |score| < 0.5 → typical

    Depends on: add_time_features, add_engagement_features.
    """
    df = df.copy()

    # --- Hard to fill: long-open + reposted + few applicants ---
    inv_apv = pd.Series(
        np.where(
            df["applications_per_vacancy"] > 0,
            1 / df["applications_per_vacancy"],
            0,
        ),
        index=df.index,
        dtype="float32",
    )

    hard = (
        _safe_zscore(df["posting_duration_days"])
        + _safe_zscore(df["metadata_repostCount"])
        + _safe_zscore(inv_apv)
    ) / 3
    df["hard_to_fill_score"] = hard.astype("float32")

    # --- Demand intensity: many applicants competing per vacancy ---
    df["demand_intensity_score"] = _safe_zscore(
        df["applications_per_vacancy"].fillna(0)
    )

    # --- Posting quality: high views, good conversion, normal duration ---
    quality = (
        _safe_zscore(df["metadata_totalNumberOfView"])
        + _safe_zscore(df["applications_per_view"].fillna(0))
        - _safe_zscore(df["posting_duration_days"])
    ) / 3
    df["posting_quality_score"] = quality.astype("float32")

    # --- Hidden gem: low views but high conversion (niche, well-targeted) ---
    view_p25 = df["metadata_totalNumberOfView"].quantile(0.25)
    apv_p75 = df["applications_per_view"].quantile(0.75)
    df["hidden_gem_flag"] = (
        (df["metadata_totalNumberOfView"] <= view_p25)
        & (df["applications_per_view"] >= apv_p75)
    )

    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_features(
    df_raw: pd.DataFrame,
    snapshot_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Run all feature blocks on a copy of the raw frame.

    Treat the input as immutable — this returns a brand-new DataFrame so you
    can re-run feature engineering without re-reading from disk.

    Order matters because later blocks depend on earlier ones:
      time -> salary -> engagement -> title -> company -> quality flags
      -> normalization -> recency -> company enrichments -> composite
    """
    df = df_raw.copy()
    df = add_time_features(df)
    df = add_salary_features(df)
    df = add_engagement_features(df)
    df = add_title_features(df)
    df = add_company_features(df)
    df = add_quality_flags(df)            # needs title_seniority
    df = add_normalized_features(df)      # needs applications_per_view
    df = add_recency_features(df, snapshot_date=snapshot_date)
    df = add_company_enrichments(df)
    df = add_composite_scores(df)         # uses many derived features
    return df


if __name__ == "__main__":
    # Larger synthetic frame so within-category benchmarks are meaningful
    n = 60
    rng = np.random.default_rng(42)
    categories = ["Information Technology", "Healthcare", "Food & Beverage",
                  "Manufacturing", "Banking & Finance"]
    companies = ["TRUST RECRUIT PTE. LTD.", "ACME INDUSTRIES PTE. LTD.",
                 "BLUEPEAK CONSULTING PTE. LTD.", "FOODCO PTE. LTD.",
                 "TECHWAVE PTE. LTD.", "CAREER STAFFING PTE. LTD."]
    titles = [
        "Senior Software Engineer", "Junior Data Analyst", "Head of Engineering",
        "Marketing Manager", "Chief Operating Officer", "Food Technologist",
        "Sales Coordinator", "Mid-level Designer", "Trainee Accountant",
        "Principal Architect", "Operations Lead", "Senior .NET Developer",
    ]
    position_levels = ["Non-executive", "Executive", "Senior Executive",
                       "Manager", "Senior Management"]

    start_dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 365, size=n), unit="D"
    )
    sal_min = rng.integers(1500, 12000, size=n).astype("int32")
    sal_max = (sal_min + rng.integers(500, 4000, size=n)).astype("int32")

    sample = pd.DataFrame({
        "employmentTypes": pd.Categorical(rng.choice(["Permanent", "Full Time"], n)),
        "metadata_expiryDate": start_dates + pd.Timedelta(days=30),
        "metadata_isPostedOnBehalf": rng.choice([False, True], n),
        "metadata_jobPostId": pd.array([f"MCF-{i}" for i in range(n)], dtype="string"),
        "metadata_newPostingDate": start_dates,
        "metadata_originalPostingDate": start_dates - pd.to_timedelta(
            rng.integers(0, 30, n), unit="D"
        ),
        "metadata_repostCount": rng.integers(0, 5, n).astype("int8"),
        "metadata_totalNumberJobApplication": rng.integers(0, 100, n).astype("int16"),
        "metadata_totalNumberOfView": rng.integers(0, 3000, n).astype("int16"),
        "minimumYearsExperience": rng.integers(0, 12, n).astype("int8"),
        "numberOfVacancies": rng.integers(1, 15, n).astype("int16"),
        "positionLevels": pd.Categorical(rng.choice(position_levels, n)),
        "postedCompany_name": pd.Categorical(rng.choice(companies, n)),
        "salary_maximum": sal_max,
        "salary_minimum": sal_min,
        "status_jobStatus": pd.Categorical(rng.choice(["Open", "Closed"], n)),
        "title": pd.array(rng.choice(titles, n), dtype="string"),
        "average_salary": ((sal_min + sal_max) / 2).astype("float32"),
        "is_hybrid": rng.choice([True, False], n),
        "category_1": pd.Categorical(rng.choice(categories, n)),
        "category_2": pd.Categorical(rng.choice(categories + [None], n)),
    })

    out = build_features(sample)

    new_cols = [c for c in out.columns if c not in sample.columns]
    print(f"Input columns:   {len(sample.columns)}")
    print(f"Output columns:  {len(out.columns)}")
    print(f"Derived columns: {len(new_cols)}\n")

    print("Derived columns and dtypes:")
    print(out[new_cols].dtypes.to_string())

    print("\nSample rows (first 3, transposed):")
    print(out[new_cols].head(3).T.to_string())
