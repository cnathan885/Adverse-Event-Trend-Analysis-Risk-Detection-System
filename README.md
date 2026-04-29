# Adverse Event Trend Analysis & Risk Detection System
### End-to-End Data Analytics Portfolio Project

---

## Overview

This project demonstrates a complete data analytics pipeline applied to medical device adverse event data (modelled after the FDA MAUDE database). It showcases skills in data acquisition, SQL/Python wrangling, statistical modelling, machine learning, and automated reporting — all commonly required in data analyst roles in regulated industries.

---

## Pipeline Architecture

```
openFDA API  ──►  Phase 1: Wrangling  ──►  Phase 2: Analysis  ──►  Phase 3: Report
(+ synthetic)    pandas + DuckDB SQL       scipy + sklearn         Jinja2 HTML
```

### Phase 1 — Data Acquisition & Wrangling (`01_data_acquisition.py`)
| Step | Technique | Skill Demonstrated |
|---|---|---|
| API ingestion | `requests` + openFDA REST | Real-world data retrieval |
| Synthetic supplement | `numpy` distributions | Understanding of data generation |
| Text field cleaning | Regex + keyword extraction | Handling messy/unstructured data |
| Date parsing & validation | `pandas` with error coercion | Data quality enforcement |
| Feature engineering | Severity score, late-report flag | Business logic translation |
| Relational storage | DuckDB with JOIN query | SQL in shared compute environments |

### Phase 2 — Analysis & Statistical Modelling (`02_analysis_modeling.py`)
| Analysis | Method | Justification |
|---|---|---|
| Spike detection | Rolling Z-score (6-mo window, ±2σ) | Flags statistically unusual months |
| Sustained shift detection | CUSUM algorithm | Catches gradual trends Z-score misses |
| Aging hypothesis | OLS regression + Pearson r | Tests directional relationship |
| Risk classification | Random Forest (200 trees, CV=5) | Handles non-linear interactions |
| Model validation | Held-out test set + AUC + cross-val | Prevents overfitting claims |

### Phase 3 — Reporting (`03_generate_report.py`)
- Jinja2 HTML template with embedded base64 figures → single-file portable report
- All KPIs and charts regenerate automatically on each pipeline run
- No manual copy-paste: fully automated insight delivery

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (all 3 phases)
chmod +x run_pipeline.sh
./run_pipeline.sh

# Or run phases individually
python 01_data_acquisition.py
python 02_analysis_modeling.py
python 03_generate_report.py
```

Open `outputs/MAUDE_Risk_Report.html` in any browser to view the report.

---

## Output Files

| File | Description |
|---|---|
| `data/cleaned_events.csv` | Cleaned adverse event records |
| `data/device_master.csv` | Device registry (relational JOIN table) |
| `data/device_summary.csv` | SQL aggregation result |
| `data/maude.duckdb` | DuckDB relational database |
| `data/analysis_results.json` | All model outputs (machine-readable) |
| `outputs/fig[1-5]_*.png` | Publication-quality figures |
| `outputs/MAUDE_Risk_Report.html` | **Self-contained HTML report** |

---

## Key Analytical Findings (from latest run)

Results in `data/analysis_results.json` — see the HTML report for full interpretation.

- **Spike detection**: Z-score + CUSUM identifies months with anomalous event volumes, enabling rapid safety signal response.
- **Hardware aging**: OLS regression tests whether older devices produce worse patient outcomes. The p-value determines if this relationship is statistically significant.
- **Risk classifier**: Random Forest AUC >0.80 demonstrates the pipeline can reliably triage incoming reports for expedited review.
- **Compliance**: Manufacturers and device types are ranked by FDA reporting timeline adherence.

---

## Tech Stack

- **Python** 3.9+ — core language
- **pandas** — data manipulation and cleaning
- **DuckDB** — embedded SQL / relational queries
- **scipy.stats** — statistical tests (Pearson correlation)
- **scikit-learn** — LinearRegression, RandomForestClassifier, cross_val_score
- **matplotlib / seaborn** — publication-quality dark-theme visualisation
- **Jinja2** — HTML report templating
- **requests** — openFDA API integration

---

## Data Sources

- **Primary**: [openFDA Device Adverse Events API](https://open.fda.gov/apis/device/event/) — real FDA MAUDE data
- **Supplementary**: High-fidelity synthetic dataset generated to ensure sufficient sample size and to inject controlled spike/long-tail patterns for analysis demonstration

---

## Project Structure

```
maude_project/
├── 01_data_acquisition.py   # Phase 1: Fetch, clean, load to DuckDB
├── 02_analysis_modeling.py  # Phase 2: Time series, regression, RF
├── 03_generate_report.py    # Phase 3: Automated HTML report
├── run_pipeline.sh          # Single-command runner
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── data/                    # Generated data artifacts (gitignored except .gitkeep)
│   └── .gitkeep
└── outputs/                 # Generated figures and final report
    └── .gitkeep
```

---

*Portfolio project — demonstrates end-to-end data analytics capabilities for data analyst roles in regulated industries.*
