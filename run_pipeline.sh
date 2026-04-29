#!/usr/bin/env bash
# ============================================================
# run_pipeline.sh — Single-command pipeline runner
# Adverse Event Trend Analysis & Risk Detection System
# ============================================================
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh
#
# Requirements: Python 3.9+ with packages in requirements.txt
# ============================================================

set -euo pipefail

BOLD="\033[1m"
GREEN="\033[32m"
BLUE="\033[34m"
YELLOW="\033[33m"
RESET="\033[0m"

echo -e "${BOLD}${BLUE}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║   MAUDE Adverse Event Risk Detection Pipeline        ║"
echo "║   Portfolio Project — Data Analyst Showcase          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${RESET}"

START=$(date +%s)

echo -e "${YELLOW}[1/3] Phase 1 — Data Acquisition & Wrangling${RESET}"
python 01_data_acquisition.py
echo -e "${GREEN}      ✓ Data ready${RESET}\n"

echo -e "${YELLOW}[2/3] Phase 2 — Statistical Analysis & Modelling${RESET}"
python 02_analysis_modeling.py
echo -e "${GREEN}      ✓ Analysis complete${RESET}\n"

echo -e "${YELLOW}[3/3] Phase 3 — Report Generation${RESET}"
python 03_generate_report.py
echo -e "${GREEN}      ✓ Report generated${RESET}\n"

END=$(date +%s)
ELAPSED=$((END - START))

echo -e "${BOLD}${GREEN}Pipeline finished in ${ELAPSED}s${RESET}"
echo -e "Report → ${BOLD}outputs/MAUDE_Risk_Report.html${RESET}"
echo ""
echo "Intermediate artifacts:"
echo "  data/cleaned_events.csv    — cleaned adverse event records"
echo "  data/device_master.csv     — device registry (JOIN table)"
echo "  data/device_summary.csv    — SQL JOIN result summary"
echo "  data/maude.duckdb          — relational DuckDB database"
echo "  data/analysis_results.json — all model outputs (machine-readable)"
echo "  outputs/fig[1-5]_*.png     — publication-quality figures"
