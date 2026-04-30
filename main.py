import io
import logging
import re
from datetime import date
import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

app = FastAPI(title="Fair Pay Pilot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MANDATORY_FIELDS = ["Salary", "Gender", "Job Level", "Department"]

COLUMN_VARIANTS = {
    "Salary": ["salary", "pay", "annual salary", "compensation", "base salary", "base pay"],
    "Gender": ["gender", "sex"],
    "Job Level": ["job level", "level", "grade", "band", "seniority", "tier"],
    "Department": ["department", "dept", "team", "division", "business unit"],
    "Race/Ethnicity": ["race/ethnicity", "ethnicity", "race", "diversity"],
    "Years at Company": ["years at company", "tenure", "years", "seniority", "length of service", "service years"],
    "Performance Rating": ["performance rating", "performance", "rating", "review", "score", "annual rating"],
    "Employee ID": ["employee id", "employee_id", "emp id", "emp_id"],
    "Job Title": ["job title", "job_title", "title", "position", "role"],
    "Bonus": ["bonus", "bonus amount", "annual bonus", "variable pay", "incentive"],
    "Total Compensation": ["total compensation", "total comp", "total pay", "total reward"],
    "Hire Date": ["hire date", "start date", "date joined", "join date"],
    "Pay Band Min": ["pay band min", "band min", "salary band min", "range min"],
    "Pay Band Max": ["pay band max", "band max", "salary band max", "range max"],
    "Pay Band Mid": ["pay band mid", "band mid", "midpoint", "salary midpoint"],
}


def _preprocess_csv_text(text: str) -> str:
    """Remove thousands-separator commas (e.g. £82,000 → £82000) before pandas sees the text.

    Without this, an unquoted value like £82,000 looks like two fields to the CSV
    parser and raises "Expected N fields, saw N+1".
    """
    return re.sub(r"(?<=\d),(?=\d)", "", text)


def resolve_columns(df_columns: list) -> dict:
    col_lower_map = {c.lower().strip(): c for c in df_columns}
    resolved = {}
    for canonical, variants in COLUMN_VARIANTS.items():
        for variant in variants:
            if variant in col_lower_map:
                resolved[canonical] = col_lower_map[variant]
                break
    return resolved


def clean_salary(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    s = s.replace("£", "").replace("$", "").replace("€", "")
    s = s.replace(" ", "")
    if not s:
        return np.nan
    if s[-1].lower() == "k":
        try:
            return float(s[:-1]) * 1000
        except ValueError:
            logger.warning("Could not parse numeric value %r — row will be skipped", value)
            return np.nan
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    try:
        return float(s)
    except ValueError:
        logger.warning("Could not parse numeric value %r — row will be skipped", value)
        return np.nan


def gap_table(avg_by_group: pd.Series) -> dict:
    highest = float(avg_by_group.max())
    result = {}
    for group, avg in avg_by_group.items():
        avg = float(avg)
        result[str(group)] = {
            "average_salary": round(avg, 2),
            "pay_gap_vs_highest_pct": round((highest - avg) / highest * 100, 2) if highest > 0 else 0.0,
        }
    return result


def adjusted_gap_table(df: pd.DataFrame, dimension: str, col: str = "Salary") -> dict:
    grouped = df.groupby(["Job Level", dimension])[col].mean()
    levels: dict = {}
    for (level, group), avg in grouped.items():
        level_key = str(level)
        if level_key not in levels:
            levels[level_key] = {}
        levels[level_key][str(group)] = round(float(avg), 2)

    result = {}
    for level_key, subgroups in levels.items():
        highest = max(subgroups.values())
        result[level_key] = {
            group: {
                "average_salary": avg,
                "pay_gap_vs_highest_pct": round((highest - avg) / highest * 100, 2) if highest > 0 else 0.0,
            }
            for group, avg in subgroups.items()
        }
    return result


# ── Regression helpers ─────────────────────────────────────────────────────────

def _convert_job_level_numeric(series: pd.Series, salary_series: pd.Series) -> pd.Series:
    """Extract digits from level strings (L1→1). Falls back to salary-rank ordering."""
    extracted = series.astype(str).str.extract(r"(\d+)", expand=False)
    numeric = pd.to_numeric(extracted, errors="coerce")
    if numeric.notna().mean() >= 0.5:
        return numeric
    medians = (
        pd.DataFrame({"level": series, "salary": salary_series})
        .groupby("level")["salary"]
        .median()
        .sort_values()
    )
    return series.map({lvl: i + 1 for i, lvl in enumerate(medians.index)})


def _encode_gender_binary(series: pd.Series) -> "pd.Series | None":
    """Female=1, Male=0. Returns None if values aren't recognisable."""
    male_labels = {"male", "m", "man", "men"}
    female_labels = {"female", "f", "woman", "women"}
    lower = series.astype(str).str.lower().str.strip()
    result = pd.Series(np.nan, index=series.index, dtype=float)
    result[lower.isin(male_labels)] = 0.0
    result[lower.isin(female_labels)] = 1.0
    if (result == 0.0).any() and (result == 1.0).any():
        return result
    return None


def _build_base_features(
    df: pd.DataFrame, has_years: bool, has_perf: bool
) -> "tuple[pd.DataFrame, list[str]]":
    parts: list = []
    controlled: list[str] = []

    jl_num = _convert_job_level_numeric(df["Job Level"], df["Salary"])
    if jl_num.nunique() > 1:
        parts.append(jl_num.rename("job_level"))
        controlled.append("Job Level")

    if df["Department"].nunique() > 1:
        dept_dummies = pd.get_dummies(df["Department"], prefix="dept", drop_first=True, dtype=float)
        parts.append(dept_dummies)
        controlled.append("Department")

    if has_years and "Years at Company" in df.columns:
        yrs = pd.to_numeric(df["Years at Company"], errors="coerce")
        if yrs.nunique() > 1:
            parts.append(yrs.rename("years_at_company"))
            controlled.append("Years at Company")

    if has_perf and "Performance Rating" in df.columns:
        perf_num = pd.to_numeric(df["Performance Rating"], errors="coerce")
        if perf_num.notna().mean() >= 0.5 and perf_num.nunique() > 1:
            parts.append(perf_num.rename("performance_rating"))
            controlled.append("Performance Rating")
        elif df["Performance Rating"].nunique() > 1:
            perf_dummies = pd.get_dummies(
                df["Performance Rating"], prefix="perf", drop_first=True, dtype=float
            )
            parts.append(perf_dummies)
            controlled.append("Performance Rating")

    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    return X, controlled


def _gender_interpretation(
    gap_pct: "float | None", p_value: float, n: int, controlled: "list[str]",
    target_label: str = "salary",
) -> str:
    if gap_pct is None:
        return f"Could not calculate the gender pay gap in {target_label}."
    controls = ", ".join(v.lower() for v in controlled) if controlled else "available factors"
    direction = "less" if gap_pct < 0 else "more"
    abs_gap = abs(round(gap_pct, 1))
    if p_value < 0.05:
        sig = "This gap is statistically significant."
    else:
        sig = "This is not statistically significant — it could be due to chance given the sample size."
    text = f"After controlling for {controls}, women earn {abs_gap}% {direction} than men in {target_label}. {sig}"
    if n < 50:
        text += " Results should be treated with caution — fewer than 50 employees in the dataset reduces statistical reliability."
    return text


def _ethnicity_interpretation(
    group: str, gap_pct: "float | None", p_value: float, n: int, reference_group: str
) -> str:
    if gap_pct is None:
        return f"Could not calculate the pay gap for {group}."
    direction = "less" if gap_pct < 0 else "more"
    abs_gap = abs(round(gap_pct, 1))
    if p_value < 0.05:
        sig = "This gap is statistically significant."
    else:
        sig = "This gap is not statistically significant."
    text = (
        f"After controlling for job level and department, {group} employees are paid "
        f"{abs_gap}% {direction} than {reference_group} employees. {sig}"
    )
    if n < 50:
        text += " Results should be treated with caution — fewer than 50 employees in the dataset reduces statistical reliability."
    return text


def _run_gender_regression_on(
    df: pd.DataFrame, target_col: str, has_years: bool, has_perf: bool
) -> dict:
    if len(df) < 20:
        return {"error": "Sample too small for regression — fewer than 20 employees in the dataset."}
    try:
        X_base, controlled = _build_base_features(df, has_years, has_perf)

        gender_enc = _encode_gender_binary(df["Gender"])
        if gender_enc is None:
            return {"error": "Could not encode Gender column — expected Male/Female values."}

        X = pd.concat([X_base, gender_enc.rename("gender_female")], axis=1)
        X = sm.add_constant(X)
        y = df[target_col]

        valid = X.notna().all(axis=1) & y.notna()
        X_fit, y_fit = X[valid], y[valid]

        if len(y_fit) < 20:
            return {"error": "Sample too small for regression after removing rows with missing values."}

        model = sm.OLS(y_fit, X_fit).fit()

        coef = float(model.params["gender_female"])
        pval = float(model.pvalues["gender_female"])
        ci = model.conf_int().loc["gender_female"]

        male_mask = df["Gender"].astype(str).str.lower().str.strip().isin({"male", "m", "man", "men"})
        avg_male_val = df.loc[valid & male_mask, target_col].mean()
        gap_pct = float(coef / avg_male_val * 100) if pd.notna(avg_male_val) and avg_male_val > 0 else None

        n = len(y_fit)
        target_label = "salary" if target_col == "Salary" else "total compensation"
        return {
            "unexplained_gender_gap_pct": round(gap_pct, 2) if gap_pct is not None else None,
            "is_significant": bool(pval < 0.05),
            "p_value": round(pval, 3),
            "confidence_interval": [round(float(ci.iloc[0]), 2), round(float(ci.iloc[1]), 2)],
            "variables_controlled": controlled,
            "sample_size_warning": n < 50,
            "interpretation": _gender_interpretation(gap_pct, pval, n, controlled, target_label),
        }
    except Exception as exc:
        return {"error": f"Regression failed: {exc}"}


def _run_ethnicity_regression_on(
    df: pd.DataFrame, target_col: str, has_years: bool, has_perf: bool
) -> dict:
    if len(df) < 20:
        return {"error": "Sample too small for regression — fewer than 20 employees in the dataset."}
    try:
        unique_groups = list(df["Race/Ethnicity"].dropna().unique())
        if len(unique_groups) < 2:
            return {"error": "Fewer than 2 ethnicity groups found — regression not applicable."}

        avg_by_eth = df.groupby("Race/Ethnicity")[target_col].mean()
        reference_group = str(avg_by_eth.idxmax())
        other_groups = [str(g) for g in unique_groups if str(g) != reference_group]

        X_base, controlled = _build_base_features(df, has_years, has_perf)

        eth_features = pd.DataFrame(index=df.index)
        col_to_group: dict = {}
        for group in other_groups:
            cname = f"eth_{group}"
            eth_features[cname] = (df["Race/Ethnicity"].astype(str) == group).astype(float)
            col_to_group[cname] = group

        X = pd.concat([X_base, eth_features], axis=1)
        X = sm.add_constant(X)
        y = df[target_col]

        valid = X.notna().all(axis=1) & y.notna() & df["Race/Ethnicity"].notna()
        X_fit, y_fit = X[valid], y[valid]

        if len(y_fit) < 20:
            return {"error": "Sample too small for regression after removing rows with missing values."}

        model = sm.OLS(y_fit, X_fit).fit()

        ref_mask = df["Race/Ethnicity"].astype(str) == reference_group
        avg_ref = float(df.loc[valid & ref_mask, target_col].mean())

        n = len(y_fit)
        groups_result: dict = {}
        for cname, group in col_to_group.items():
            if cname not in model.params.index:
                continue
            coef = float(model.params[cname])
            pval = float(model.pvalues[cname])
            ci = model.conf_int().loc[cname]
            gap_pct = float(coef / avg_ref * 100) if avg_ref > 0 else None
            groups_result[group] = {
                "unexplained_gap_vs_highest_paid_pct": round(gap_pct, 2) if gap_pct is not None else None,
                "is_significant": bool(pval < 0.05),
                "p_value": round(pval, 3),
                "confidence_interval": [round(float(ci.iloc[0]), 2), round(float(ci.iloc[1]), 2)],
                "sample_size_warning": n < 50,
                "interpretation": _ethnicity_interpretation(group, gap_pct, pval, n, reference_group),
            }

        return {
            "reference_group": reference_group,
            "variables_controlled": controlled,
            "sample_size_warning": n < 50,
            "groups": groups_result,
        }
    except Exception as exc:
        return {"error": f"Regression failed: {exc}"}


def run_gender_regression(df: pd.DataFrame, has_years: bool, has_perf: bool) -> dict:
    return _run_gender_regression_on(df, "Salary", has_years, has_perf)


def run_ethnicity_regression(df: pd.DataFrame, has_years: bool, has_perf: bool) -> dict:
    return _run_ethnicity_regression_on(df, "Salary", has_years, has_perf)


# ── New analysis builders ──────────────────────────────────────────────────────

def build_bonus_analysis(df: pd.DataFrame, has_race_ethnicity: bool) -> dict:
    df = df.copy()
    df["Bonus"] = pd.to_numeric(df["Bonus"].apply(clean_salary), errors="coerce").fillna(0)

    by_gender_mean = df.groupby("Gender")["Bonus"].mean()
    participation = df.groupby("Gender").apply(lambda x: (x["Bonus"] > 0).mean() * 100)
    highest = float(by_gender_mean.max())

    by_gender_result: dict = {}
    for gender in by_gender_mean.index:
        avg = float(by_gender_mean[gender])
        by_gender_result[str(gender)] = {
            "average_bonus": round(avg, 2),
            "bonus_gap_vs_highest_pct": round((highest - avg) / highest * 100, 2) if highest > 0 else 0.0,
            "participation_rate_pct": round(float(participation.get(gender, 0)), 2),
        }

    by_level_gender = df.groupby(["Job Level", "Gender"])["Bonus"].mean()
    by_job_level: dict = {}
    for (level, gender), avg in by_level_gender.items():
        lk = str(level)
        if lk not in by_job_level:
            by_job_level[lk] = {}
        by_job_level[lk][str(gender)] = round(float(avg), 2)

    result: dict = {"by_gender": by_gender_result, "by_job_level_and_gender": by_job_level}

    if has_race_ethnicity:
        by_race_mean = df.groupby("Race/Ethnicity")["Bonus"].mean()
        highest_race = float(by_race_mean.max())
        result["by_race_ethnicity"] = {
            str(g): {
                "average_bonus": round(float(v), 2),
                "bonus_gap_vs_highest_pct": round((highest_race - float(v)) / highest_race * 100, 2)
                if highest_race > 0 else 0.0,
            }
            for g, v in by_race_mean.items()
        }

    return result


def build_total_comp_analysis(
    df: pd.DataFrame, has_race_ethnicity: bool, has_years: bool, has_perf: bool, has_bonus: bool
) -> dict:
    df = df.copy()
    df["Total Compensation"] = pd.to_numeric(
        df["Total Compensation"].apply(clean_salary), errors="coerce"
    )
    df_tc = df.dropna(subset=["Total Compensation"]).copy()

    if df_tc.empty:
        return {"error": "No valid Total Compensation values found after cleaning."}

    unadjusted: dict = {
        "by_gender": gap_table(df_tc.groupby("Gender")["Total Compensation"].mean()),
    }
    adjusted: dict = {
        "by_gender_within_job_level": adjusted_gap_table(df_tc, "Gender", "Total Compensation"),
    }

    if has_race_ethnicity:
        unadjusted["by_race_ethnicity"] = gap_table(
            df_tc.groupby("Race/Ethnicity")["Total Compensation"].mean()
        )
        adjusted["by_race_within_job_level"] = adjusted_gap_table(
            df_tc, "Race/Ethnicity", "Total Compensation"
        )

    bonus_note = "Bonus data is available but was not included as a control variable in this regression."

    gender_reg = _run_gender_regression_on(df_tc, "Total Compensation", has_years, has_perf)
    if has_bonus and "error" not in gender_reg:
        gender_reg["note"] = bonus_note
    regression: dict = {"gender_regression": gender_reg}

    if has_race_ethnicity:
        eth_reg = _run_ethnicity_regression_on(df_tc, "Total Compensation", has_years, has_perf)
        if has_bonus and "error" not in eth_reg:
            eth_reg["note"] = bonus_note
        regression["ethnicity_regression"] = eth_reg

    return {"unadjusted_gap": unadjusted, "adjusted_gap": adjusted, "regression": regression}


def build_compa_ratio_analysis(df: pd.DataFrame, has_race_ethnicity: bool) -> dict:
    df = df.copy()
    for col in ("Pay Band Min", "Pay Band Max", "Pay Band Mid"):
        df[col] = pd.to_numeric(df[col].apply(clean_salary), errors="coerce")

    df["_compa_ratio"] = df["Salary"] / df["Pay Band Mid"] * 100

    dq_count = int(((df["_compa_ratio"] < 30) | (df["_compa_ratio"] > 300)).sum())
    df_cr = df[df["_compa_ratio"].between(30, 300)].copy()

    if df_cr.empty:
        return {"error": "No valid compa-ratio values found after data quality filtering."}

    by_gender_mean = df_cr.groupby("Gender")["_compa_ratio"].mean()
    below_90 = df_cr.groupby("Gender").apply(lambda x: (x["_compa_ratio"] < 90).mean() * 100)
    above_110 = df_cr.groupby("Gender").apply(lambda x: (x["_compa_ratio"] > 110).mean() * 100)

    by_gender_result: dict = {}
    for gender in by_gender_mean.index:
        by_gender_result[str(gender)] = {
            "average_compa_ratio": round(float(by_gender_mean[gender]), 2),
            "pct_below_90": round(float(below_90.get(gender, 0)), 2),
            "pct_above_110": round(float(above_110.get(gender, 0)), 2),
        }

    by_level_gender = df_cr.groupby(["Job Level", "Gender"])["_compa_ratio"].mean()
    by_job_level: dict = {}
    for (level, gender), avg in by_level_gender.items():
        lk = str(level)
        if lk not in by_job_level:
            by_job_level[lk] = {}
        by_job_level[lk][str(gender)] = round(float(avg), 2)

    at_risk: list = []
    for _, row in df[df["_compa_ratio"] < 80].iterrows():
        entry: dict = {
            "department": str(row["Department"]),
            "job_level": str(row["Job Level"]),
            "gender": str(row["Gender"]),
            "salary": round(float(row["Salary"]), 2),
            "compa_ratio": round(float(row["_compa_ratio"]), 2),
            "flag": "at risk of underpay",
        }
        if "Employee ID" in df.columns:
            entry["employee_id"] = str(row["Employee ID"])
        if "Race/Ethnicity" in df.columns:
            entry["race_ethnicity"] = str(row["Race/Ethnicity"])
        at_risk.append(entry)

    result: dict = {
        "by_gender": by_gender_result,
        "by_job_level_and_gender": by_job_level,
        "at_risk_of_underpay": at_risk,
    }

    if has_race_ethnicity:
        by_race = df_cr.groupby("Race/Ethnicity")["_compa_ratio"].mean()
        result["by_race_ethnicity"] = {str(k): round(float(v), 2) for k, v in by_race.items()}

    if dq_count > 0:
        result["data_quality_warning"] = (
            f"{dq_count} employee(s) had compa-ratio values outside the expected range (30–300) "
            "and were excluded from this analysis."
        )

    return result


def build_intersectionality_analysis(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["_intersect"] = df["Gender"].astype(str) + " - " + df["Race/Ethnicity"].astype(str)

    agg = df.groupby("_intersect")["Salary"].agg(average_salary="mean", headcount="count")
    valid_agg = agg[agg["headcount"] >= 3]
    highest = float(valid_agg["average_salary"].max()) if not valid_agg.empty else 0.0

    result: dict = {}
    for group, row in agg.iterrows():
        count = int(row["headcount"])
        if count < 3:
            result[str(group)] = {"result": "Insufficient data", "headcount": count}
        else:
            avg = float(row["average_salary"])
            result[str(group)] = {
                "average_salary": round(avg, 2),
                "headcount": count,
                "pay_gap_vs_highest_pct": round((highest - avg) / highest * 100, 2) if highest > 0 else 0.0,
            }

    return result


def build_starting_salary_analysis(df: pd.DataFrame, has_race_ethnicity: bool) -> dict:
    df = df.copy()
    df["_hire_date"] = pd.to_datetime(df["Hire Date"], errors="coerce")
    still_null = df["_hire_date"].isna() & df["Hire Date"].notna()
    if still_null.any():
        df.loc[still_null, "_hire_date"] = pd.to_datetime(
            df.loc[still_null, "Hire Date"], dayfirst=True, errors="coerce"
        )

    df["_hire_year"] = df["_hire_date"].dt.year
    max_hire_date = df["_hire_date"].max()
    if pd.isna(max_hire_date):
        return {"note": "No valid hire dates found in the dataset."}
    cutoff_year = max_hire_date.year - 3
    recent = df[df["_hire_year"] >= cutoff_year].copy()

    if recent.empty:
        return {"note": f"No employees hired since {cutoff_year} found in the dataset."}

    level_avg = df.groupby("Job Level")["Salary"].mean()

    def group_summary(gdf: pd.DataFrame) -> dict:
        flagged: list = []
        for level, ldf in gdf.groupby("Job Level"):
            avg_sal = float(ldf["Salary"].mean())
            level_overall = level_avg.get(level)
            if level_overall is not None and float(level_overall) > 0:
                pct_diff = (avg_sal - float(level_overall)) / float(level_overall) * 100
                if pct_diff < -5:
                    flagged.append({
                        "job_level": str(level),
                        "group_average_salary": round(avg_sal, 2),
                        "level_overall_average_salary": round(float(level_overall), 2),
                        "pct_below_level_average": round(abs(pct_diff), 2),
                    })
        return {
            "average_starting_salary": round(float(gdf["Salary"].mean()), 2),
            "headcount": int(len(gdf)),
            "flagged_below_level_average": flagged,
        }

    result: dict = {
        "recent_hire_cutoff_year": int(cutoff_year),
        "recent_hire_count": int(len(recent)),
        "by_gender": {str(g): group_summary(gdf) for g, gdf in recent.groupby("Gender")},
    }

    if has_race_ethnicity:
        result["by_race_ethnicity"] = {
            str(g): group_summary(gdf) for g, gdf in recent.groupby("Race/Ethnicity")
        }

    return result


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    raw = await file.read()
    try:
        csv_text = _preprocess_csv_text(raw.decode("utf-8"))
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}")

    # Trim whitespace from all string columns and drop completely empty rows
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    df = df.dropna(how="all").copy()

    # Resolve CSV columns to canonical names (case-insensitive, variant-aware)
    col_map = resolve_columns(list(df.columns))

    missing_mandatory = [f for f in MANDATORY_FIELDS if f not in col_map]
    if missing_mandatory:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "CSV is missing required columns.",
                "missing_fields": missing_mandatory,
                "hint": {
                    "Salary": "accepted names: salary, pay, compensation, annual salary, base salary, base pay",
                    "Gender": "accepted names: gender, sex",
                    "Job Level": "accepted names: job level, level, grade, band, seniority",
                    "Department": "accepted names: department, dept, team, division",
                },
            },
        )

    # ── Step 1: rename all columns to canonical names BEFORE any data parsing ────
    df = df.rename(columns={v: k for k, v in col_map.items()})

    # Record the original CSV column name that was mapped to Salary for diagnostics
    original_salary_col = col_map.get("Salary", "Salary")
    logger.info("Salary column mapped from original CSV column %r", original_salary_col)

    has_employee_id = "Employee ID" in df.columns
    has_job_title = "Job Title" in df.columns
    has_race_ethnicity = "Race/Ethnicity" in df.columns
    has_years_at_company = "Years at Company" in df.columns
    has_performance_rating = "Performance Rating" in df.columns
    has_bonus = "Bonus" in df.columns
    has_total_comp = "Total Compensation" in df.columns
    has_hire_date = "Hire Date" in df.columns
    has_pay_band_min = "Pay Band Min" in df.columns
    has_pay_band_max = "Pay Band Max" in df.columns
    has_pay_band_mid = "Pay Band Mid" in df.columns
    has_compa_ratio = has_pay_band_min and has_pay_band_max and has_pay_band_mid

    # ── Step 2: parse and clean the salary column (always "Salary" after mapping) ─
    # Diagnostic: if the column is full of non-numeric values it was mapped wrongly
    salary_probe = df["Salary"].dropna().head(20)
    if not salary_probe.empty:
        probe_cleaned = salary_probe.apply(clean_salary)
        if pd.to_numeric(probe_cleaned, errors="coerce").notna().sum() == 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Salary column appears to contain non-numeric data. "
                    f"Detected column: '{original_salary_col}'. "
                    "Please check your column mapping."
                ),
            )

    original_salaries = df["Salary"].copy()
    df["Salary"] = df["Salary"].apply(clean_salary)
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")

    failed_mask = df["Salary"].isna() & original_salaries.notna()
    failed_examples = [str(v) for v in original_salaries[failed_mask].head(5).tolist()]

    df = df.dropna(subset=["Salary"]).copy()

    if len(df) < 3:
        detail = "No valid salary rows found in the CSV." if df.empty else "Fewer than 3 valid salary rows found in the CSV."
        if failed_examples:
            detail += f" Example values that failed to parse: {', '.join(failed_examples)}."
        raise HTTPException(status_code=400, detail=detail)

    # ── available_analyses ────────────────────────────────────────────────────
    skipped_reasons: dict = {}
    if not has_race_ethnicity:
        skipped_reasons["ethnicity_gap"] = "Race/Ethnicity column not found"
    if not has_years_at_company:
        skipped_reasons["tenure_analysis"] = "Years at Company column not found"
    if not has_performance_rating:
        skipped_reasons["performance_analysis"] = "Performance Rating column not found"
    if not has_employee_id:
        skipped_reasons["outlier_tracking"] = "Employee ID column not found"
    if not has_bonus:
        skipped_reasons["bonus_analysis"] = "Bonus column not found"
    if not has_total_comp:
        skipped_reasons["total_comp_analysis"] = "Total Compensation column not found"
    if not has_compa_ratio:
        skipped_reasons["compa_ratio_analysis"] = "Pay Band Min, Max and Mid columns not all present"
    if not has_hire_date:
        skipped_reasons["starting_salary_analysis"] = "Hire Date column not found"

    available_analyses: dict = {
        "gender_gap": True,
        "department_breakdown": True,
        "ethnicity_gap": has_race_ethnicity,
        "tenure_analysis": has_years_at_company,
        "performance_analysis": has_performance_rating,
        "outlier_tracking": has_employee_id,
        "bonus_analysis": has_bonus,
        "total_comp_analysis": has_total_comp,
        "compa_ratio_analysis": has_compa_ratio,
        "intersectionality_analysis": has_race_ethnicity,
        "starting_salary_analysis": has_hire_date,
        "skipped_reasons": skipped_reasons,
    }

    # ── Gender pay gap ────────────────────────────────────────────────────────
    unadjusted_pay_gap: dict = {
        "by_gender": gap_table(df.groupby("Gender")["Salary"].mean()),
    }
    adjusted_pay_gap: dict = {
        "by_gender_within_job_level": adjusted_gap_table(df, "Gender"),
    }

    # ── Race/Ethnicity pay gap (optional) ────────────────────────────────────
    if has_race_ethnicity:
        unadjusted_pay_gap["by_race_ethnicity"] = gap_table(
            df.groupby("Race/Ethnicity")["Salary"].mean()
        )
        adjusted_pay_gap["by_race_within_job_level"] = adjusted_gap_table(df, "Race/Ethnicity")

    # ── Tenure analysis (optional) ────────────────────────────────────────────
    tenure_analysis = None
    if has_years_at_company:
        df["Years at Company"] = pd.to_numeric(df["Years at Company"], errors="coerce")
        bins = [0, 1, 3, 5, 10, float("inf")]
        labels = ["<1 year", "1-3 years", "3-5 years", "5-10 years", "10+ years"]
        df["_tenure_band"] = pd.cut(df["Years at Company"], bins=bins, labels=labels, right=False)
        agg = df.groupby("_tenure_band", observed=True)["Salary"].agg(
            average_salary="mean", headcount="count"
        )
        tenure_analysis = {
            str(band): {
                "average_salary": round(float(row["average_salary"]), 2),
                "headcount": int(row["headcount"]),
            }
            for band, row in agg.iterrows()
        }

    # ── Performance vs pay analysis (optional) ────────────────────────────────
    performance_analysis = None
    if has_performance_rating:
        agg = df.groupby("Performance Rating")["Salary"].agg(
            average_salary="mean", headcount="count"
        )
        performance_analysis = {
            str(rating): {
                "average_salary": round(float(row["average_salary"]), 2),
                "headcount": int(row["headcount"]),
            }
            for rating, row in agg.iterrows()
        }

    # ── Outlier detection ─────────────────────────────────────────────────────
    df["_peer_avg"] = df.groupby("Job Level")["Salary"].transform("mean")
    df["_ratio"] = df["Salary"] / df["_peer_avg"]
    flagged_rows = df[(df["_ratio"] > 1.20) | (df["_ratio"] < 0.80)]

    flagged_outliers: list = []
    for _, row in flagged_rows.iterrows():
        gender_val = row["Gender"]
        entry: dict = {
            "department": str(row["Department"]),
            "job_level": str(row["Job Level"]),
            "gender": "Not specified" if (pd.isna(gender_val) or str(gender_val).strip().lower() in {"", "nan", "none", "null"}) else str(gender_val),
            "salary": round(float(row["Salary"]), 2),
            "peer_group_average": round(float(row["_peer_avg"]), 2),
            "variance_pct": round((float(row["_ratio"]) - 1) * 100, 2),
            "flag": "ABOVE average" if row["_ratio"] > 1.20 else "BELOW average",
        }
        if has_employee_id:
            entry["employee_id"] = str(row["Employee ID"])
        if has_job_title:
            entry["job_title"] = str(row["Job Title"])
        if has_race_ethnicity:
            entry["race_ethnicity"] = str(row["Race/Ethnicity"])
        flagged_outliers.append(entry)

    # ── Department summary ────────────────────────────────────────────────────
    dept_gender = (
        df.groupby(["Department", "Gender"])["Salary"]
        .agg(average_salary="mean", headcount="count")
        .reset_index()
    )
    department_summary: dict = {}
    for _, row in dept_gender.iterrows():
        dept = str(row["Department"])
        gender = str(row["Gender"])
        if dept not in department_summary:
            department_summary[dept] = {"by_gender": {}, "total": {}}
        department_summary[dept]["by_gender"][gender] = {
            "average_salary": round(float(row["average_salary"]), 2),
            "headcount": int(row["headcount"]),
        }

    dept_totals = df.groupby("Department")["Salary"].agg(average_salary="mean", headcount="count")
    for dept, row in dept_totals.iterrows():
        dept_key = str(dept)
        if dept_key in department_summary:
            department_summary[dept_key]["total"] = {
                "average_salary": round(float(row["average_salary"]), 2),
                "headcount": int(row["headcount"]),
            }

    # ── Summary ───────────────────────────────────────────────────────────────
    summary: dict = {
        "total_employees": len(df),
        "total_departments": int(df["Department"].nunique()),
        "total_job_levels": int(df["Job Level"].nunique()),
        "overall_average_salary": round(float(df["Salary"].mean()), 2),
        "overall_median_salary": round(float(df["Salary"].median()), 2),
        "flagged_outlier_count": len(flagged_outliers),
    }

    response: dict = {
        "available_analyses": available_analyses,
        "summary": summary,
        "unadjusted_pay_gap": unadjusted_pay_gap,
        "adjusted_pay_gap": adjusted_pay_gap,
        "flagged_outliers": flagged_outliers,
        "department_summary": department_summary,
    }

    if tenure_analysis is not None:
        response["tenure_analysis"] = tenure_analysis
    if performance_analysis is not None:
        response["performance_analysis"] = performance_analysis

    # ── New optional analyses ─────────────────────────────────────────────────
    if has_bonus:
        response["bonus_analysis"] = build_bonus_analysis(df, has_race_ethnicity)

    if has_total_comp:
        response["total_comp_analysis"] = build_total_comp_analysis(
            df, has_race_ethnicity, has_years_at_company, has_performance_rating, has_bonus
        )

    if has_compa_ratio:
        response["compa_ratio_analysis"] = build_compa_ratio_analysis(df, has_race_ethnicity)

    if has_race_ethnicity:
        response["intersectionality_analysis"] = build_intersectionality_analysis(df)

    if has_hire_date:
        response["starting_salary_analysis"] = build_starting_salary_analysis(df, has_race_ethnicity)

    # ── Regression analysis ───────────────────────────────────────────────────
    bonus_note = "Bonus data is available but was not included as a control variable in this regression."

    gender_reg = run_gender_regression(df, has_years_at_company, has_performance_rating)
    if has_bonus and "error" not in gender_reg:
        gender_reg["note"] = bonus_note
    regression_analysis: dict = {"gender_regression": gender_reg}

    if has_race_ethnicity:
        eth_reg = run_ethnicity_regression(df, has_years_at_company, has_performance_rating)
        if has_bonus and "error" not in eth_reg:
            eth_reg["note"] = bonus_note
        regression_analysis["ethnicity_regression"] = eth_reg

    response["regression_analysis"] = regression_analysis

    return response
