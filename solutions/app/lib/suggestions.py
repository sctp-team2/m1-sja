"""Per-posting recommendation engine.

`suggest_for_row(row)` evaluates the rule set against a single posting
and returns a list of suggestion dicts. `suggest_for_df(df)` runs the
vectorised equivalent across a frame and returns a long-form table that
Page 3 Section 3.5 ("Recruiter action list") aggregates.

Each suggestion references the feature that triggered it so recruiters
can audit the reasoning. Rules and messages mirror spec section 8.
"""

from __future__ import annotations

import pandas as pd


SEVERITY_BADGE = {"High": "🔴", "Medium": "🟡", "Low": "🔵"}

# Centralised rule set — keep in sync with the spec.
RULES = [
    "compensation_bottom_quintile",
    "hard_to_fill",
    "low_conversion_listing",
    "seniority_mismatch",
    "high_applicant_volume",
    "reposted_stale",
    "salary_undisclosed",
]


def _category_apv_p75(df: pd.DataFrame) -> pd.Series:
    """Return category_1 → p75 of applications_per_vacancy."""
    return (
        df.groupby("category_1", observed=True)["applications_per_vacancy"]
        .quantile(0.75)
    )


def _category_median_salary(df: pd.DataFrame) -> pd.Series:
    return (
        df.groupby("category_1", observed=True)["average_salary"]
        .median()
    )


def suggest_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every rule across `df` and return a long-form suggestion table.

    Columns: row_index, severity, type, message, rule, category_1,
    title, postedCompany_name.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["row_index", "severity", "type", "message", "rule",
                     "category_1", "title", "postedCompany_name"]
        )

    apv_p75 = _category_apv_p75(df)
    cat_median_salary = _category_median_salary(df)

    pieces = []

    # 1. Compensation — bottom 20% within category
    mask = df["salary_percentile_within_category"] < 0.2
    if mask.any():
        sub = df[mask].copy()
        medians = sub["category_1"].map(cat_median_salary)
        sub["message"] = [
            f"Pay is in bottom 20% for {cat}. Median in this category is S${med:,.0f}."
            for cat, med in zip(sub["category_1"].astype(str), medians.fillna(0))
        ]
        pieces.append(_emit(sub, "High", "compensation",
                            "compensation_bottom_quintile"))

    # 2. Hard to fill
    mask = df["hard_to_fill_score"] > 2.0
    if mask.any():
        sub = df[mask].copy()
        sub["message"] = (
            "Above-market difficulty to fill (hard_to_fill_score="
            + sub["hard_to_fill_score"].round(2).astype(str)
            + "). Consider direct sourcing or broader requirements."
        )
        pieces.append(_emit(sub, "Medium", "sourcing", "hard_to_fill"))

    # 3. Low-conversion listing — eyeballs without applications
    mask = (df["posting_quality_score"] < -1.5) & (~df["zero_engagement_flag"])
    if mask.any():
        sub = df[mask].copy()
        sub["message"] = (
            "Posting attracts views but underperforms on conversion "
            f"(posting_quality_score < -1.5). Review JD specificity."
        )
        pieces.append(_emit(sub, "Medium", "listing", "low_conversion_listing"))

    # 4. Seniority mismatch
    mask = df["seniority_mismatch_flag"]
    if mask.any():
        sub = df[mask].copy()
        sub["message"] = [
            f"Title seniority ({ts}) doesn't match position level ({pl}). "
            "Verify intended level."
            for ts, pl in zip(sub["title_seniority"].astype(str),
                               sub["positionLevels"].astype(str))
        ]
        pieces.append(_emit(sub, "Low", "categorization", "seniority_mismatch"))

    # 5. High applicant volume — above category p75
    cat_p75 = df["category_1"].map(apv_p75)
    mask = df["applications_per_vacancy"] > cat_p75
    mask = mask.fillna(False)
    if mask.any():
        sub = df[mask].copy()
        sub["message"] = (
            "High applicant volume vs category peers "
            f"(applications_per_vacancy above p75). Tighten shortlist criteria."
        )
        pieces.append(_emit(sub, "Low", "shortlisting", "high_applicant_volume"))

    # 6. Reposted and long-open
    mask = df["is_reposted"] & (df["posting_duration_days"] > 60)
    if mask.any():
        sub = df[mask].copy()
        sub["message"] = (
            "Reposted and open >60 days "
            f"(posting_duration_days, metadata_repostCount). "
            "Reassess JD or compensation."
        )
        pieces.append(_emit(sub, "Medium", "strategy", "reposted_stale"))

    # 7. Salary undisclosed — defensive; ~0% on this dataset
    mask = df["salary_undisclosed_flag"]
    if mask.any():
        sub = df[mask].copy()
        sub["message"] = (
            "Salary not disclosed (salary_undisclosed_flag). Postings with "
            "ranges typically get higher application rates."
        )
        pieces.append(_emit(sub, "Low", "transparency", "salary_undisclosed"))

    if not pieces:
        return pd.DataFrame(
            columns=["row_index", "severity", "type", "message", "rule",
                     "category_1", "title", "postedCompany_name"]
        )
    return pd.concat(pieces, ignore_index=True)


def _emit(sub: pd.DataFrame, severity: str, suggestion_type: str,
          rule: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "row_index": sub.index,
        "severity": severity,
        "type": suggestion_type,
        "message": sub["message"].values,
        "rule": rule,
        "category_1": sub["category_1"].astype(str).values,
        "title": sub["title"].astype(str).values,
        "postedCompany_name": sub["postedCompany_name"].astype(str).values,
    })
    return out


def suggest_for_row(row: pd.Series) -> list[dict]:
    """Evaluate a single posting. Thin wrapper over `suggest_for_df`.

    Useful for one-off drill-ins (e.g., "show me suggestions for this
    promising-role table row"). For bulk work always use `suggest_for_df`.
    """
    df = row.to_frame().T
    out = suggest_for_df(df)
    return out[["severity", "type", "message", "rule"]].to_dict("records")


# Aggregated rule summaries for Page 3 Section 3.5
ACTION_LIST_LABELS = {
    "compensation_bottom_quintile": "{n} roles need compensation review (pay in bottom 20% of category)",
    "hard_to_fill": "{n} roles flagged as hard-to-fill (score > 2.0)",
    "low_conversion_listing": "{n} roles with low conversion despite high views",
    "seniority_mismatch": "{n} roles with title/seniority mismatch",
    "high_applicant_volume": "{n} roles with high applicant volume (above category p75)",
    "reposted_stale": "{n} reposted roles open >60 days",
    "salary_undisclosed": "{n} roles with undisclosed salary",
}
