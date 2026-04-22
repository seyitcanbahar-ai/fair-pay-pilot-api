import io
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

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
    "Salary": ["salary", "pay", "compensation", "annual salary", "base salary", "base pay"],
    "Gender": ["gender", "sex"],
    "Job Level": ["job level", "level", "grade", "band", "seniority"],
    "Department": ["department", "dept", "team", "division"],
    "Race/Ethnicity": ["race/ethnicity", "ethnicity", "race", "diversity"],
    "Years at Company": ["years at company", "tenure", "years"],
    "Performance Rating": ["performance rating", "performance", "rating", "review"],
    "Employee ID": ["employee id", "employee_id", "emp id", "emp_id"],
    "Job Title": ["job title", "job_title", "title", "position", "role"],
}


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
        return value
    return str(value).replace("£", "").replace("$", "").replace("€", "").replace(",", "").strip()


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


def adjusted_gap_table(df: pd.DataFrame, dimension: str) -> dict:
    grouped = df.groupby(["Job Level", dimension])["Salary"].mean()
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


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    raw = await file.read()
    try:
        df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
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

    # Rename columns to canonical names
    df = df.rename(columns={v: k for k, v in col_map.items()})

    has_employee_id = "Employee ID" in df.columns
    has_job_title = "Job Title" in df.columns
    has_race_ethnicity = "Race/Ethnicity" in df.columns
    has_years_at_company = "Years at Company" in df.columns
    has_performance_rating = "Performance Rating" in df.columns

    # Clean and validate salary
    df["Salary"] = df["Salary"].apply(clean_salary)
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df = df.dropna(subset=["Salary"]).copy()

    if df.empty:
        raise HTTPException(status_code=400, detail="No valid salary rows found in the CSV.")

    # ── available_analyses ────────────────────────────────────────────────────
    skipped_reasons = {}
    if not has_race_ethnicity:
        skipped_reasons["ethnicity_gap"] = "Race/Ethnicity column not found"
    if not has_years_at_company:
        skipped_reasons["tenure_analysis"] = "Years at Company column not found"
    if not has_performance_rating:
        skipped_reasons["performance_analysis"] = "Performance Rating column not found"
    if not has_employee_id:
        skipped_reasons["outlier_tracking"] = "Employee ID column not found"

    available_analyses = {
        "gender_gap": True,
        "department_breakdown": True,
        "ethnicity_gap": has_race_ethnicity,
        "tenure_analysis": has_years_at_company,
        "performance_analysis": has_performance_rating,
        "outlier_tracking": has_employee_id,
        "skipped_reasons": skipped_reasons,
    }

    # ── Gender pay gap ────────────────────────────────────────────────────────
    unadjusted_pay_gap = {
        "by_gender": gap_table(df.groupby("Gender")["Salary"].mean()),
    }
    adjusted_pay_gap = {
        "by_gender_within_job_level": adjusted_gap_table(df, "Gender"),
    }

    # ── Race/Ethnicity pay gap (optional) ─────────────────────────────────────
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

    flagged_outliers = []
    for _, row in flagged_rows.iterrows():
        entry: dict = {
            "department": str(row["Department"]),
            "job_level": str(row["Job Level"]),
            "gender": str(row["Gender"]),
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
    summary = {
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

    return response
