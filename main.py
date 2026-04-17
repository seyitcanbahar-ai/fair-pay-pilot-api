import io
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Fair Pay Pilot API")

# Allow any frontend (including Lovable) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUIRED_COLUMNS = [
    "Employee ID",
    "Full Name",
    "Department",
    "Job Title",
    "Job Level",
    "Salary",
    "Gender",
    "Race/Ethnicity",
    "Years at Company",
    "Performance Rating",
]


def gap_table(avg_by_group: pd.Series) -> dict:
    """
    Given a Series of {group_name: average_salary}, return a dict with
    the average salary and the pay gap % vs the highest-paid group.
    A gap of 0 % means this group IS the highest-paid reference group.
    """
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
    """
    Calculate average salary by Job Level × dimension (e.g. Gender or Race/Ethnicity).
    Within each job level, compute the pay gap vs the highest-paid sub-group.
    """
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
    # ── 1. Read and validate the uploaded file ────────────────────────────────
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    raw = await file.read()
    try:
        df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing these required columns: {missing_cols}",
        )

    # Clean salary — drop rows where salary is not a number
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df = df.dropna(subset=["Salary"]).copy()

    if df.empty:
        raise HTTPException(status_code=400, detail="No valid salary rows found in the CSV.")

    # ── 2. Unadjusted pay gap ─────────────────────────────────────────────────
    unadjusted = {
        "by_gender": gap_table(df.groupby("Gender")["Salary"].mean()),
        "by_race_ethnicity": gap_table(df.groupby("Race/Ethnicity")["Salary"].mean()),
    }

    # ── 3. Adjusted pay gap (controls for Job Level) ──────────────────────────
    adjusted = {
        "by_gender_within_job_level": adjusted_gap_table(df, "Gender"),
        "by_race_within_job_level": adjusted_gap_table(df, "Race/Ethnicity"),
    }

    # ── 4. Flagged outliers ───────────────────────────────────────────────────
    # Peer group = employees in the same Job Level.
    # Flag anyone paid more than 20 % above or below their peer group average.
    df["_peer_avg"] = df.groupby("Job Level")["Salary"].transform("mean")
    df["_ratio"] = df["Salary"] / df["_peer_avg"]

    outlier_mask = (df["_ratio"] > 1.20) | (df["_ratio"] < 0.80)
    flagged_rows = df[outlier_mask]

    flagged_outliers = [
        {
            "employee_id": str(row["Employee ID"]),
            "full_name": str(row["Full Name"]),
            "department": str(row["Department"]),
            "job_title": str(row["Job Title"]),
            "job_level": str(row["Job Level"]),
            "gender": str(row["Gender"]),
            "race_ethnicity": str(row["Race/Ethnicity"]),
            "salary": round(float(row["Salary"]), 2),
            "peer_group_average": round(float(row["_peer_avg"]), 2),
            "variance_pct": round((float(row["_ratio"]) - 1) * 100, 2),
            "flag": "ABOVE average" if row["_ratio"] > 1.20 else "BELOW average",
        }
        for _, row in flagged_rows.iterrows()
    ]

    # ── 5. Department summary ─────────────────────────────────────────────────
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

    # ── 6. High-level summary ─────────────────────────────────────────────────
    summary = {
        "total_employees": len(df),
        "total_departments": int(df["Department"].nunique()),
        "total_job_levels": int(df["Job Level"].nunique()),
        "overall_average_salary": round(float(df["Salary"].mean()), 2),
        "overall_median_salary": round(float(df["Salary"].median()), 2),
        "flagged_outlier_count": len(flagged_outliers),
    }

    return {
        "summary": summary,
        "unadjusted_pay_gap": unadjusted,
        "adjusted_pay_gap": adjusted,
        "flagged_outliers": flagged_outliers,
        "department_summary": department_summary,
    }
