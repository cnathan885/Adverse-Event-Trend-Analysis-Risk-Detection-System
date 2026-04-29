"""
Phase 1: Data Acquisition & Wrangling
======================================
Fetches FDA MAUDE adverse event reports via the openFDA API,
generates realistic synthetic supplementary data, and loads
everything into a DuckDB relational database.

Author: Portfolio Project — Adverse Event Trend Analysis & Risk Detection System
"""

import requests
import pandas as pd
import numpy as np
import json
import duckdb
import os
import re
import time
import random
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = "https://api.fda.gov/device/event.json"
DB_PATH  = "data/maude.duckdb"
SEED     = 42
np.random.seed(SEED)
random.seed(SEED)

DEVICE_CLASSES = {
    "Infusion Pump":    {"class": "II",  "base_failure_rate": 0.15, "aging_factor": 0.08},
    "Ventilator":       {"class": "III", "base_failure_rate": 0.08, "aging_factor": 0.12},
    "Cardiac Monitor":  {"class": "II",  "base_failure_rate": 0.10, "aging_factor": 0.07},
    "Insulin Pump":     {"class": "III", "base_failure_rate": 0.12, "aging_factor": 0.09},
    "Defibrillator":    {"class": "III", "base_failure_rate": 0.06, "aging_factor": 0.15},
    "Surgical Robot":   {"class": "III", "base_failure_rate": 0.04, "aging_factor": 0.05},
    "MRI Scanner":      {"class": "II",  "base_failure_rate": 0.05, "aging_factor": 0.06},
    "Pacemaker":        {"class": "III", "base_failure_rate": 0.07, "aging_factor": 0.11},
    "Blood Glucose":    {"class": "II",  "base_failure_rate": 0.18, "aging_factor": 0.04},
    "Dialysis Machine": {"class": "II",  "base_failure_rate": 0.09, "aging_factor": 0.10},
}

FAILURE_MODES = [
    "software malfunction", "battery failure", "sensor drift",
    "mechanical wear", "electrical short", "firmware crash",
    "calibration error", "display failure", "alarm failure",
    "connectivity loss", "overheating", "power supply failure",
]

PATIENT_OUTCOMES = [
    "no injury", "minor injury", "moderate injury",
    "serious injury", "hospitalization", "death", "unknown",
]

OUTCOME_WEIGHTS = [0.35, 0.25, 0.18, 0.12, 0.06, 0.02, 0.02]


# ── Step 1: Fetch real FDA MAUDE data ─────────────────────────────────────────

def fetch_fda_maude(limit: int = 500) -> pd.DataFrame:
    """Pull adverse event records from openFDA and return a tidy DataFrame."""
    print("Fetching FDA MAUDE records…")
    all_records = []
    skip = 0
    batch = 100

    while len(all_records) < limit:
        params = {
            "limit": min(batch, limit - len(all_records)),
            "skip":  skip,
            "search": "date_received:[20200101+TO+20241231]",
        }
        try:
            r = requests.get(BASE_URL, params=params, timeout=15)
            if r.status_code != 200:
                print(f"  API returned {r.status_code} — switching to synthetic data.")
                return pd.DataFrame()
            data = r.json()
            results = data.get("results", [])
            if not results:
                break
            all_records.extend(results)
            skip += batch
            time.sleep(0.3)
            print(f"  Fetched {len(all_records)} records…", end="\r")
        except Exception as e:
            print(f"  Network error: {e} — switching to synthetic data.")
            return pd.DataFrame()

    print(f"\n  Done. {len(all_records)} FDA records retrieved.")

    rows = []
    for rec in all_records:
        try:
            device = rec.get("device", [{}])[0]
            mdr    = rec.get("mdr_text", [{}])[0]
            rows.append({
                "report_id":        rec.get("mdr_report_key", ""),
                "date_received":    rec.get("date_received", ""),
                "date_of_event":    rec.get("date_of_event", ""),
                "device_name":      device.get("generic_name", "Unknown Device"),
                "brand_name":       device.get("brand_name", ""),
                "manufacturer":     device.get("manufacturer_d_name", ""),
                "model_number":     device.get("model_number", ""),
                "device_age_years": np.nan,
                "failure_mode":     mdr.get("text", "")[:200] if mdr else "",
                "patient_outcome":  rec.get("patient", [{}])[0].get("sequence_number_outcome", ["unknown"])[0]
                                    if rec.get("patient") else "unknown",
                "report_type":      rec.get("report_source_code", ""),
                "event_type":       rec.get("event_type", ""),
                "days_to_report":   np.nan,
                "source":           "FDA_API",
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


# ── Step 2: Generate synthetic data ───────────────────────────────────────────

def generate_synthetic_events(n: int = 3000) -> pd.DataFrame:
    """
    Create a realistic synthetic dataset that mirrors the FDA MAUDE schema.
    Injects two 'spikes' and a long-tail risk pattern for the analysis phases.
    """
    print(f"Generating {n} synthetic adverse event records…")

    start_date = datetime(2020, 1, 1)
    end_date   = datetime(2024, 12, 31)
    date_range = (end_date - start_date).days

    rows = []
    device_names = list(DEVICE_CLASSES.keys())

    for i in range(n):
        device_name = random.choices(
            device_names,
            weights=[10, 6, 9, 8, 5, 3, 4, 6, 12, 7],
            k=1,
        )[0]
        meta = DEVICE_CLASSES[device_name]

        # Spike injection: Ventilator spike in mid-2021, Insulin Pump spike in Q3-2023
        base_days = random.randint(0, date_range)
        if device_name == "Ventilator" and random.random() < 0.35:
            base_days = random.randint(365, 550)   # mid-2021 window
        if device_name == "Insulin Pump" and random.random() < 0.30:
            base_days = random.randint(1277, 1460) # Q3-2023 window

        event_date    = start_date + timedelta(days=base_days)
        days_to_report = max(1, int(np.random.exponential(scale=18)))
        received_date  = event_date + timedelta(days=days_to_report)

        device_age   = max(0.1, np.random.gamma(shape=2.5, scale=1.8))
        failure_prob = meta["base_failure_rate"] + meta["aging_factor"] * device_age

        failure_mode = random.choices(
            FAILURE_MODES,
            weights=[12, 10, 9, 11, 7, 8, 9, 6, 8, 7, 6, 7],
            k=1,
        )[0]

        # Long-tail: old devices with sensor drift → worse outcomes
        if device_age > 6 and failure_mode == "sensor drift":
            outcome = random.choices(
                PATIENT_OUTCOMES,
                weights=[0.10, 0.15, 0.20, 0.28, 0.18, 0.07, 0.02],
                k=1,
            )[0]
        else:
            outcome = random.choices(PATIENT_OUTCOMES, weights=OUTCOME_WEIGHTS, k=1)[0]

        # Inject messy text to demonstrate wrangling skills
        messy_descriptions = [
            f"Device experienced {failure_mode}. Patient outcome: {outcome}. Lot# {random.randint(10000,99999)}",
            f"FAILURE TYPE: {failure_mode.upper()}  --  outcome={outcome}  age={device_age:.1f}yrs",
            f"{failure_mode}; reported by facility. n/a outcome unknown",
            f"Pt outcome: {outcome}. Mode of failure - {failure_mode} (confirmed)",
            f"  {failure_mode}  ,  {outcome}  ",  # whitespace mess
            f"{failure_mode} failure noted. See attached. Outcome: {outcome}.",
        ]

        rows.append({
            "report_id":        f"SYN-{i+1:06d}",
            "date_of_event":    event_date.strftime("%Y%m%d"),
            "date_received":    received_date.strftime("%Y%m%d"),
            "device_name":      device_name,
            "brand_name":       f"{device_name} Pro {random.choice(['X','S','Elite','Plus','Ultra'])}",
            "manufacturer":     random.choice([
                "MedTech Corp", "BioDevice Inc", "LifeCare Systems",
                "PrecisionMed", "ClinTech USA", "Apex Medical",
            ]),
            "model_number":     f"MDL-{random.randint(1000,9999)}",
            "device_age_years": round(device_age, 2),
            "failure_mode":     random.choice(messy_descriptions),
            "patient_outcome":  outcome,
            "report_type":      random.choice(["Manufacturer", "Voluntary", "Mandatory"]),
            "event_type":       random.choice(["Malfunction", "Injury", "Death", "No Apparent Injury"]),
            "days_to_report":   days_to_report,
            "source":           "SYNTHETIC",
        })

    df = pd.DataFrame(rows)
    print(f"  Generated {len(df)} records.")
    return df


# ── Step 3: Data Cleaning ─────────────────────────────────────────────────────

def clean_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demonstrate real-world wrangling:
    - Standardise dates
    - Strip messy text fields
    - Extract structured failure mode from free text
    - Handle nulls and outliers
    - Flag late reports (regulatory compliance metric)
    """
    print("Cleaning and transforming events…")
    original_len = len(df)

    # Parse dates
    df["date_of_event"]  = pd.to_datetime(df["date_of_event"],  format="%Y%m%d", errors="coerce")
    df["date_received"]  = pd.to_datetime(df["date_received"],  format="%Y%m%d", errors="coerce")

    # Drop records with unparseable event dates
    df = df.dropna(subset=["date_of_event"])

    # Standardise device name casing
    df["device_name"] = df["device_name"].str.strip().str.title()

    # Clean outcome column: lowercase, strip punctuation, map aliases
    outcome_map = {
        "1": "death", "2": "serious injury", "3": "hospitalization",
        "4": "moderate injury", "5": "minor injury", "6": "no injury",
        "na": "unknown", "n/a": "unknown", "": "unknown",
    }
    df["patient_outcome"] = (
        df["patient_outcome"]
          .str.lower()
          .str.strip()
          .replace(outcome_map)
    )

    # Extract clean failure mode keyword from messy free-text
    failure_keywords = [
        "software malfunction", "battery failure", "sensor drift",
        "mechanical wear", "electrical short", "firmware crash",
        "calibration error", "display failure", "alarm failure",
        "connectivity loss", "overheating", "power supply failure",
    ]
    def extract_failure_mode(text: str) -> str:
        if not isinstance(text, str):
            return "unknown"
        text_lower = text.lower()
        for kw in failure_keywords:
            if kw in text_lower:
                return kw
        return "other"

    df["failure_mode_raw"]   = df["failure_mode"]
    df["failure_mode_clean"] = df["failure_mode"].apply(extract_failure_mode)

    # Recalculate days_to_report from parsed dates (overrides noisy column)
    df["days_to_report"] = (df["date_received"] - df["date_of_event"]).dt.days
    df["days_to_report"] = df["days_to_report"].clip(lower=0, upper=365)

    # Late reporting flag (FDA requires reports within 30 days for mandatory reporters)
    df["late_report"] = (df["days_to_report"] > 30).astype(int)

    # Outcome severity score (for regression target)
    severity_map = {
        "death": 5, "serious injury": 4, "hospitalization": 3,
        "moderate injury": 2, "minor injury": 1, "no injury": 0, "unknown": np.nan,
    }
    df["severity_score"] = df["patient_outcome"].map(severity_map)

    # Add time columns for trend analysis
    df["year"]       = df["date_of_event"].dt.year
    df["month"]      = df["date_of_event"].dt.month
    df["year_month"] = df["date_of_event"].dt.to_period("M")

    # Flag device age outliers (> 15 yrs = data quality flag)
    df["age_flag"] = (df["device_age_years"] > 15).astype(int)

    removed = original_len - len(df)
    print(f"  Cleaning complete. Removed {removed} invalid records. Final: {len(df)} rows.")
    print(f"  Late reports: {df['late_report'].sum()} ({df['late_report'].mean()*100:.1f}%)")
    print(f"  Unknown failure modes: {(df['failure_mode_clean']=='other').sum()}")
    return df


# ── Step 4: Device History (separate table for SQL JOIN demo) ──────────────────

def generate_device_history(event_df: pd.DataFrame) -> pd.DataFrame:
    """Simulate a Device Master Record table — joined later in SQL."""
    print("Generating device history / master records…")
    devices = []
    for device_name, meta in DEVICE_CLASSES.items():
        for _ in range(random.randint(8, 20)):
            manufacture_year = random.randint(2010, 2022)
            devices.append({
                "device_name":         device_name.title(),
                "model_number":        f"MDL-{random.randint(1000,9999)}",
                "fda_class":           meta["class"],
                "manufacture_year":    manufacture_year,
                "expected_lifespan_yrs": random.randint(5, 12),
                "last_recall_year":    random.choice([None, None, None, 2019, 2021, 2022, 2023]),
                "software_version":    f"{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,9)}",
                "regulatory_clearance":random.choice(["510(k)", "PMA", "De Novo", "HDE"]),
            })
    df = pd.DataFrame(devices)
    df.insert(0, "device_id", [f"DEV-{i+1:04d}" for i in range(len(df))])
    print(f"  {len(df)} device master records created.")
    return df


# ── Step 5: Load into DuckDB ───────────────────────────────────────────────────

def load_to_duckdb(events_df: pd.DataFrame, devices_df: pd.DataFrame, db_path: str):
    """Persist both tables to DuckDB and run a validation JOIN query."""
    print(f"Loading data into DuckDB at '{db_path}'…")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    con = duckdb.connect(db_path)

    # Drop and recreate
    con.execute("DROP TABLE IF EXISTS adverse_events")
    con.execute("DROP TABLE IF EXISTS device_master")

    con.execute("""
        CREATE TABLE adverse_events AS
        SELECT * FROM events_df
    """)
    con.execute("""
        CREATE TABLE device_master AS
        SELECT * FROM devices_df
    """)

    # Validation JOIN (demonstrates relational DB skill)
    result = con.execute("""
        SELECT
            ae.device_name,
            dm.fda_class,
            dm.regulatory_clearance,
            COUNT(*)                            AS event_count,
            ROUND(AVG(ae.severity_score), 3)    AS avg_severity,
            ROUND(AVG(ae.device_age_years), 2)  AS avg_age_yrs,
            SUM(ae.late_report)                 AS late_reports,
            ROUND(AVG(ae.days_to_report), 1)    AS avg_days_to_report
        FROM adverse_events ae
        LEFT JOIN device_master dm
            ON ae.device_name = dm.device_name
        GROUP BY ae.device_name, dm.fda_class, dm.regulatory_clearance
        ORDER BY event_count DESC
    """).fetchdf()

    print("\n── Device Summary (SQL JOIN result) ──────────────────────────────")
    print(result.to_string(index=False))
    result.to_csv("data/device_summary.csv", index=False)

    count = con.execute("SELECT COUNT(*) FROM adverse_events").fetchone()[0]
    print(f"\n  DuckDB loaded: {count} adverse event rows")
    con.close()
    print("  Database saved.\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 1: Data Acquisition & Wrangling")
    print("=" * 60)

    # Try FDA API first; fall back to synthetic
    events_df = fetch_fda_maude(limit=500)
    if events_df.empty or len(events_df) < 100:
        print("Using fully synthetic dataset.")
        events_df = generate_synthetic_events(n=3000)
    else:
        # Supplement real data with synthetic to boost sample size
        synthetic = generate_synthetic_events(n=2500)
        events_df = pd.concat([events_df, synthetic], ignore_index=True)
        print(f"Combined dataset: {len(events_df)} records (real + synthetic)")

    events_df  = clean_events(events_df)
    devices_df = generate_device_history(events_df)

    # Save cleaned CSVs
    events_df.to_csv("data/cleaned_events.csv", index=False)
    devices_df.to_csv("data/device_master.csv", index=False)
    print("  Saved cleaned_events.csv and device_master.csv")

    load_to_duckdb(events_df, devices_df, DB_PATH)
    print("Phase 1 complete.\n")


if __name__ == "__main__":
    main()
