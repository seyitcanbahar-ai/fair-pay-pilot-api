# Fair Pay Pilot API

A Python API that analyses employee pay equity data and returns structured JSON results.

---

## What you need installed first

- **Python 3.10 or higher** — download from https://www.python.org/downloads/
  - During installation, tick the box that says **"Add Python to PATH"**

Open the **Command Prompt** (search "cmd" in the Start menu) for all steps below.

---

## Setup (one-time)

**1. Go into the project folder**

```
cd C:\Users\Seyit\fair-pay-pilot-api
```

**2. Create a virtual environment** (keeps dependencies tidy)

```
python -m venv venv
```

**3. Activate the virtual environment**

```
venv\Scripts\activate
```

You will see `(venv)` appear at the start of the line — that means it worked.

**4. Install the dependencies**

```
pip install -r requirements.txt
```

---

## Running the API

Every time you want to start the server, run these two commands:

```
venv\Scripts\activate
uvicorn main:app --reload
```

You should see output like:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

The API is now live at **http://localhost:8000**.

To stop the server, press `Ctrl + C`.

---

## Testing the API manually

Once running, open your browser and go to:

```
http://localhost:8000/docs
```

This opens an interactive page where you can upload a CSV and see the results — no code needed.

---

## CSV format

Your CSV file must have exactly these column headers (spelling and capitalisation matters):

| Column | Example value |
|---|---|
| Employee ID | E001 |
| Full Name | Jane Smith |
| Department | Engineering |
| Job Title | Software Engineer |
| Job Level | L3 |
| Salary | 95000 |
| Gender | Female |
| Race/Ethnicity | Asian |
| Years at Company | 4 |
| Performance Rating | 3.8 |

---

## What the API returns

The `/analyze` endpoint returns a JSON object with five sections:

| Section | What it contains |
|---|---|
| `summary` | Total headcount, overall average salary, count of flagged employees |
| `unadjusted_pay_gap` | Average salary by gender and by race/ethnicity across the whole company, with pay gap % vs the highest-paid group |
| `adjusted_pay_gap` | Same breakdown but calculated within each Job Level, so it controls for seniority |
| `flagged_outliers` | Individual employees paid more than 20 % above or below their Job Level peer group average |
| `department_summary` | Average salary and headcount per department, split by gender |

---

## Calling the API from your Lovable frontend

The endpoint to call is:

```
POST http://localhost:8000/analyze
```

Send the CSV as `multipart/form-data` with the field name `file`.

CORS is enabled for all origins, so Lovable can call it without any extra configuration.

When you deploy this API to a hosting service, replace `localhost:8000` with your live URL.
