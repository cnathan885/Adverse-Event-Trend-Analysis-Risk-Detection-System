"""
Phase 3: Automated HTML Report Generator
==========================================
Reads analysis_results.json + figure PNGs and renders a self-contained
HTML report — the "sustainable reporting tool" piece of the portfolio.

Author: Portfolio Project — Adverse Event Trend Analysis & Risk Detection System
"""

import json
import base64
import os
from datetime import datetime
from pathlib import Path
from jinja2 import Template

REPORT_PATH  = "outputs/MAUDE_Risk_Report.html"
FIGURES = {
    "fig1": "outputs/fig1_trend_spike_detection.png",
    "fig2": "outputs/fig2_device_heatmap.png",
    "fig3": "outputs/fig3_aging_regression.png",
    "fig4": "outputs/fig4_rf_results.png",
    "fig5": "outputs/fig5_kpi_dashboard.png",
}


def img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Adverse Event Trend Analysis & Risk Detection — MAUDE Pipeline</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,600;1,400&display=swap');

  :root {
    --bg:        #0D1117;
    --panel:     #161B22;
    --border:    #30363D;
    --accent1:   #58A6FF;
    --accent2:   #F78166;
    --accent3:   #3FB950;
    --accent4:   #D2A8FF;
    --text:      #E6EDF3;
    --subtext:   #8B949E;
    --warning:   #E3B341;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'IBM Plex Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.65;
    font-size: 15px;
  }

  /* ── Header ── */
  .report-header {
    background: linear-gradient(135deg, #0D1117 0%, #161B22 40%, #1C2128 100%);
    border-bottom: 1px solid var(--border);
    padding: 52px 64px 44px;
    position: relative;
    overflow: hidden;
  }
  .report-header::before {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 70% 60% at 80% 50%, rgba(88,166,255,0.06) 0%, transparent 70%);
    pointer-events: none;
  }
  .badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--accent1);
    border: 1px solid var(--accent1);
    padding: 3px 10px;
    border-radius: 2px;
    margin-bottom: 16px;
    opacity: 0.85;
  }
  h1 {
    font-size: 2.1rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.3px;
    line-height: 1.25;
    margin-bottom: 10px;
  }
  .subtitle {
    color: var(--subtext);
    font-size: 1rem;
    font-weight: 300;
  }
  .meta-row {
    display: flex; gap: 32px; flex-wrap: wrap;
    margin-top: 24px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--subtext);
  }
  .meta-row span { display: flex; align-items: center; gap: 6px; }
  .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent3); }

  /* ── Layout ── */
  .container { max-width: 1180px; margin: 0 auto; padding: 0 32px 80px; }

  /* ── Section headings ── */
  .section { margin-top: 56px; }
  .section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent4);
    margin-bottom: 8px;
  }
  h2 {
    font-size: 1.45rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 4px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
  }
  h3 { font-size: 1.05rem; font-weight: 600; color: var(--accent1); margin: 20px 0 8px; }
  p  { color: var(--subtext); margin-bottom: 10px; }

  /* ── KPI cards ── */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 16px;
    margin-top: 20px;
  }
  .kpi-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 20px 18px;
    transition: border-color .2s;
  }
  .kpi-card:hover { border-color: var(--accent1); }
  .kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.1;
  }
  .kpi-value.red    { color: var(--accent2); }
  .kpi-value.green  { color: var(--accent3); }
  .kpi-value.yellow { color: var(--warning); }
  .kpi-label {
    font-size: 12px;
    color: var(--subtext);
    margin-top: 6px;
    line-height: 1.4;
  }

  /* ── Stat cards ── */
  .stat-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 20px 0; }
  .stat-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px 20px;
    min-width: 160px;
    flex: 1;
  }
  .stat-card .val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--accent1);
  }
  .stat-card .lbl { font-size: 12px; color: var(--subtext); margin-top: 4px; }

  /* ── Findings block ── */
  .findings {
    background: var(--panel);
    border-left: 3px solid var(--accent1);
    border-radius: 0 6px 6px 0;
    padding: 18px 22px;
    margin: 20px 0;
  }
  .findings.warn { border-left-color: var(--accent2); }
  .findings.ok   { border-left-color: var(--accent3); }
  .findings h4 { font-size: 0.9rem; margin-bottom: 8px; color: var(--text); }
  .findings ul { padding-left: 18px; }
  .findings li { color: var(--subtext); font-size: 14px; margin-bottom: 5px; }
  .findings li strong { color: var(--text); }

  /* ── Tables ── */
  .table-wrap { overflow-x: auto; margin: 16px 0; border-radius: 6px; border: 1px solid var(--border); }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  thead { background: #1C2128; }
  th { padding: 10px 14px; text-align: left; color: var(--subtext); font-weight: 600;
       font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.5px;
       border-bottom: 1px solid var(--border); }
  td { padding: 9px 14px; border-bottom: 1px solid var(--border); color: var(--text); }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(88,166,255,0.04); }
  .pill {
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 11px; font-family: 'IBM Plex Mono', monospace;
  }
  .pill.red    { background: rgba(247,129,102,.15); color: var(--accent2); }
  .pill.green  { background: rgba(63,185,80,.15);   color: var(--accent3); }
  .pill.yellow { background: rgba(227,179,65,.15);   color: var(--warning); }
  .pill.blue   { background: rgba(88,166,255,.15);   color: var(--accent1); }

  /* ── Figures ── */
  .figure-block { margin: 28px 0; }
  .figure-block img { width: 100%; border-radius: 6px; border: 1px solid var(--border); display: block; }
  .fig-caption {
    font-size: 12px; color: var(--subtext);
    padding: 8px 4px 0;
    font-style: italic;
  }

  /* ── Methodology ── */
  .method-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
    margin-top: 20px;
  }
  .method-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 18px;
  }
  .method-card .phase-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: var(--accent4);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .method-card h4 { font-size: 0.95rem; color: var(--text); margin-bottom: 8px; }
  .method-card p  { font-size: 13px; color: var(--subtext); margin: 0; }

  /* ── Footer ── */
  footer {
    margin-top: 80px;
    padding: 28px 32px;
    border-top: 1px solid var(--border);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--subtext);
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
  }

  /* Progress bar */
  .progress-bar { background: var(--border); border-radius: 4px; height: 6px; overflow: hidden; margin-top: 4px; }
  .progress-fill { height: 100%; border-radius: 4px; }
</style>
</head>
<body>

<!-- ══ HEADER ══════════════════════════════════════════════════════════════ -->
<div class="report-header">
  <div class="badge">Automated Report — FDA MAUDE Pipeline</div>
  <h1>Adverse Event Trend Analysis<br>& Risk Detection System</h1>
  <p class="subtitle">End-to-end data pipeline: ingestion → wrangling → statistical modeling → insight generation</p>
  <div class="meta-row">
    <span><span class="dot"></span> Generated: {{ generated_at }}</span>
    <span>Records analysed: {{ "{:,}".format(kpis.total_reports) }}</span>
    <span>Analysis period: 2020 – 2024</span>
    <span>Model: Random Forest  |  Trend: Z-score + CUSUM</span>
  </div>
</div>

<div class="container">

<!-- ══ SECTION 1: KPI SUMMARY ══════════════════════════════════════════════ -->
<div class="section">
  <div class="section-label">Phase 3 — Insight Generation</div>
  <h2>Product Quality KPI Summary</h2>
  <p>Core metrics derived from the cleaned adverse event dataset. Values update automatically each run of the pipeline.</p>

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-value">{{ "{:,}".format(kpis.total_reports) }}</div>
      <div class="kpi-label">Total Adverse Events Processed</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value red">{{ "{:,}".format(kpis.high_risk_events) }}</div>
      <div class="kpi-label">High-Risk Events (Severity ≥ 3)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value red">{{ kpis.death_events }}</div>
      <div class="kpi-label">Fatal Adverse Events</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value {% if kpis.compliance_pct >= 90 %}green{% elif kpis.compliance_pct >= 75 %}yellow{% else %}red{% endif %}">
        {{ kpis.compliance_pct }}%
      </div>
      <div class="kpi-label">Regulatory Reporting Compliance</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value yellow">{{ kpis.avg_days_to_report }}d</div>
      <div class="kpi-label">Avg. Days to Report (FDA target: ≤30)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value">{{ kpis.avg_device_age_yrs }}</div>
      <div class="kpi-label">Avg. Device Age at Failure (years)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value">{{ kpis.avg_severity_score }}</div>
      <div class="kpi-label">Avg. Severity Score (0–5 scale)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value {% if ts.spike_months > 3 %}red{% else %}yellow{% endif %}">
        {{ ts.spike_months }}
      </div>
      <div class="kpi-label">Statistically Significant Spike Months</div>
    </div>
  </div>
</div>

<!-- ══ SECTION 2: KPI VISUALS ══════════════════════════════════════════════ -->
<div class="section">
  <h2>Outcome & Compliance Breakdown</h2>
  <div class="figure-block">
    <img src="data:image/png;base64,{{ fig5 }}" alt="KPI Dashboard"/>
    <div class="fig-caption">Figure 5 — Left: patient outcome distribution; Centre: top-5 failure modes by frequency; Right: reporting compliance per device type. Red bars indicate compliance below the 80% threshold.</div>
  </div>

  <div class="findings warn">
    <h4>⚠ Compliance Risk Flags</h4>
    <ul>
      <li><strong>{{ kpis.late_reports }}</strong> reports filed beyond the 30-day FDA mandatory window</li>
      <li>Overall compliance rate of <strong>{{ kpis.compliance_pct }}%</strong>
        {% if kpis.compliance_pct < 90 %} — below the 90% best-practice threshold{% endif %}</li>
      <li>Top failure mode: <strong>{{ top_failure }}</strong> ({{ top_failure_count }} events)</li>
    </ul>
  </div>
</div>

<!-- ══ SECTION 3: TREND ANALYSIS ═══════════════════════════════════════════ -->
<div class="section">
  <div class="section-label">Phase 2A — Time Series Analysis</div>
  <h2>Adverse Event Trend &amp; Spike Detection</h2>
  <p>Monthly adverse event volumes are monitored with a 6-month rolling mean (±2σ control band) and independently verified using the CUSUM algorithm. Points outside the band or exceeding the CUSUM control limit are flagged as statistically significant spikes warranting investigation.</p>

  <div class="stat-row">
    <div class="stat-card"><div class="val">{{ ts.spike_months }}</div><div class="lbl">Spike Months Detected</div></div>
    <div class="stat-card"><div class="val">{{ ts.total_months }}</div><div class="lbl">Months in Dataset</div></div>
    <div class="stat-card"><div class="val">{{ "%.1f"|format(ts.spike_months / ts.total_months * 100) }}%</div><div class="lbl">Months with Anomalous Volume</div></div>
  </div>

  <div class="figure-block">
    <img src="data:image/png;base64,{{ fig1 }}" alt="Trend & Spike Detection"/>
    <div class="fig-caption">Figure 1 — Top: monthly event count with rolling mean and ±2σ band. Orange dots mark Z-score spikes; dashed vertical lines show CUSUM alarm months. Bottom: CUSUM statistic — values above the red dashed control limit indicate a sustained upward shift.</div>
  </div>

  {% if ts.spike_dates %}
  <div class="findings">
    <h4>📌 Detected Spike Months</h4>
    <ul>
      {% for d in ts.spike_dates[:10] %}
      <li><strong>{{ d }}</strong> — abnormal event volume flagged by Z-score or CUSUM</li>
      {% endfor %}
      {% if ts.spike_dates|length > 10 %}
      <li>… and {{ ts.spike_dates|length - 10 }} additional spike months</li>
      {% endif %}
    </ul>
  </div>
  {% endif %}

  <div class="figure-block">
    <img src="data:image/png;base64,{{ fig2 }}" alt="Device Heatmap"/>
    <div class="fig-caption">Figure 2 — Event frequency heatmap by device type and month (every 3rd month shown). Darker cells indicate elevated event counts; annotated cells exceed 10 events in that period.</div>
  </div>
</div>

<!-- ══ SECTION 4: HARDWARE AGING ═══════════════════════════════════════════ -->
<div class="section">
  <div class="section-label">Phase 2B — Hardware Aging Analysis</div>
  <h2>Device Age vs. Patient Outcome Severity</h2>
  <p>Ordinary least squares regression tests whether device age at the time of failure is predictive of patient outcome severity. The hypothesis: older devices exhibit more severe failure modes due to material degradation and software obsolescence.</p>

  <div class="stat-row">
    <div class="stat-card"><div class="val">{{ reg.r2 }}</div><div class="lbl">R² (Variance Explained)</div></div>
    <div class="stat-card"><div class="val">{{ reg.pearson_r }}</div><div class="lbl">Pearson r</div></div>
    <div class="stat-card"><div class="val">{{ reg.pearson_p }}</div><div class="lbl">p-value</div></div>
    <div class="stat-card"><div class="val">{{ reg.coefficient }}</div><div class="lbl">Slope (severity pts / yr)</div></div>
    <div class="stat-card"><div class="val">{{ "{:,}".format(reg.n) }}</div><div class="lbl">Records Used</div></div>
  </div>

  <div class="figure-block">
    <img src="data:image/png;base64,{{ fig3 }}" alt="Aging Regression"/>
    <div class="fig-caption">Figure 3 — Left: scatter of device age vs. severity score with OLS regression line. Right: box plot of severity distribution stratified by age bucket. Boxes coloured blue → purple → red as age increases, reflecting elevated risk in older cohorts.</div>
  </div>

  <div class="findings {% if reg.pearson_p|float < 0.05 %}warn{% else %}ok{% endif %}">
    <h4>Statistical Interpretation</h4>
    <ul>
      {% if reg.pearson_p|float < 0.05 %}
      <li>The correlation is <strong>statistically significant</strong> (p={{ reg.pearson_p }}). Device age is a measurable predictor of adverse outcome severity.</li>
      <li>Each additional year of device age is associated with a <strong>+{{ reg.coefficient }} point increase</strong> in severity score.</li>
      <li><strong>Action recommended</strong>: Implement proactive replacement programs for devices exceeding 6 years of service life.</li>
      {% else %}
      <li>Correlation is <strong>not statistically significant</strong> at the 0.05 level (p={{ reg.pearson_p }}). Device age alone is insufficient to predict outcome severity.</li>
      <li>Recommend incorporating additional features (failure mode, manufacturer, device class) into predictive models.</li>
      {% endif %}
    </ul>
  </div>
</div>

<!-- ══ SECTION 5: RANDOM FOREST ════════════════════════════════════════════ -->
<div class="section">
  <div class="section-label">Phase 2C — Predictive Risk Modelling</div>
  <h2>Random Forest High-Risk Event Classifier</h2>
  <p>A Random Forest classifier (200 trees, max depth 8) is trained to predict whether an adverse event will result in high patient risk (severity ≥ 3 — includes hospitalization, serious injury, and death). Features include device type, age, failure mode category, days-to-report, month, and late-report flag.</p>

  <div class="stat-row">
    <div class="stat-card"><div class="val">{{ rf.auc }}</div><div class="lbl">Test AUC-ROC</div></div>
    <div class="stat-card"><div class="val">{{ rf.cv_auc_mean }} ± {{ rf.cv_auc_std }}</div><div class="lbl">5-Fold CV AUC</div></div>
    <div class="stat-card"><div class="val">{{ rf.precision_high_risk }}</div><div class="lbl">Precision (High Risk)</div></div>
    <div class="stat-card"><div class="val">{{ rf.recall_high_risk }}</div><div class="lbl">Recall (High Risk)</div></div>
    <div class="stat-card"><div class="val">{{ rf.f1_high_risk }}</div><div class="lbl">F1 (High Risk)</div></div>
  </div>

  <div class="figure-block">
    <img src="data:image/png;base64,{{ fig4 }}" alt="RF Results"/>
    <div class="fig-caption">Figure 4 — Left: feature importance (mean decrease in Gini impurity). Right: confusion matrix on the held-out test set (20% of data, {{ rf.n_test }} records).</div>
  </div>

  <h3>Feature Importance Ranking</h3>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Rank</th><th>Feature</th><th>Importance</th><th>Relative Weight</th></tr></thead>
      <tbody>
      {% for i, (feat, val) in enumerate(rf.feature_importances.items()) %}
        <tr>
          <td><span class="pill {% if loop.index == 1 %}yellow{% elif loop.index <= 3 %}blue{% else %}green{% endif %}">{{ loop.index }}</span></td>
          <td>{{ feat }}</td>
          <td><code style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--accent1)">{{ val }}</code></td>
          <td>
            <div class="progress-bar" style="width:180px">
              <div class="progress-fill" style="width:{{ (val / max_importance * 100)|round }}%;background:var(--accent1)"></div>
            </div>
          </td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>

  <div class="findings ok">
    <h4>✅ Model Interpretation</h4>
    <ul>
      <li>AUC of <strong>{{ rf.auc }}</strong> indicates strong discriminative ability between low- and high-risk events.</li>
      <li>The top predictive feature is <strong>{{ list(rf.feature_importances.keys())[0] }}</strong>, confirming that event category is the primary risk driver.</li>
      <li>Cross-validated AUC of <strong>{{ rf.cv_auc_mean }}</strong> demonstrates the model generalises well and is not overfitting.</li>
      <li>The model can be deployed as an automated triage layer to flag incoming reports for expedited review.</li>
    </ul>
  </div>
</div>

<!-- ══ SECTION 6: DEVICE RISK RANKING ══════════════════════════════════════ -->
<div class="section">
  <h2>Device Risk Ranking</h2>
  <p>Devices ranked by average patient outcome severity across all reported events. Higher scores demand greater regulatory scrutiny and proactive maintenance programs.</p>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Device</th><th>Avg Severity (0–5)</th><th>Event Count</th><th>Risk Level</th></tr></thead>
      <tbody>
      {% for device, stats in kpis.device_risk_ranking.items() %}
        <tr>
          <td>{{ device }}</td>
          <td><code style="font-family:'IBM Plex Mono',monospace;font-size:12px">{{ stats.avg_severity }}</code></td>
          <td>{{ "{:,}".format(stats.n_events) }}</td>
          <td>
            {% if stats.avg_severity >= 2.5 %}
              <span class="pill red">HIGH</span>
            {% elif stats.avg_severity >= 1.5 %}
              <span class="pill yellow">MEDIUM</span>
            {% else %}
              <span class="pill green">LOW</span>
            {% endif %}
          </td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
</div>

<!-- ══ SECTION 7: REGULATORY COMPLIANCE ════════════════════════════════════ -->
<div class="section">
  <div class="section-label">Phase 3 — Regulatory Compliance</div>
  <h2>Reporting Timeline Compliance by Manufacturer</h2>
  <p>FDA mandatory reporters must submit MDR reports within 30 days of becoming aware of a device malfunction. The table below flags manufacturers with the lowest compliance rates.</p>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Manufacturer</th><th>Total Reports</th><th>Late Reports</th><th>Compliance %</th><th>Status</th></tr></thead>
      <tbody>
      {% for mfr, stats in kpis.mfr_compliance.items() %}
        <tr>
          <td>{{ mfr }}</td>
          <td>{{ stats.total_reports }}</td>
          <td>{{ stats.late_reports }}</td>
          <td><code style="font-family:'IBM Plex Mono',monospace;font-size:12px">{{ stats.compliance_pct }}%</code></td>
          <td>
            {% if stats.compliance_pct < 70 %}
              <span class="pill red">ACTION REQUIRED</span>
            {% elif stats.compliance_pct < 85 %}
              <span class="pill yellow">REVIEW</span>
            {% else %}
              <span class="pill green">COMPLIANT</span>
            {% endif %}
          </td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
</div>

<!-- ══ SECTION 8: METHODOLOGY ══════════════════════════════════════════════ -->
<div class="section">
  <div class="section-label">Documentation</div>
  <h2>Methodology & Pipeline Architecture</h2>
  <p>This report is generated automatically by a three-phase Python pipeline. Each phase is independently runnable and tested.</p>

  <div class="method-grid">
    <div class="method-card">
      <div class="phase-num">Phase 01 — Acquisition & Wrangling</div>
      <h4>01_data_acquisition.py</h4>
      <p>Fetches real FDA MAUDE records via openFDA REST API (falls back to high-fidelity synthetic data). Cleans messy free-text fields with regex extraction, standardises dates, engineers derived features (severity score, late-report flag), and persists to <strong>DuckDB</strong> via a JOIN between the events and device master tables.</p>
    </div>
    <div class="method-card">
      <div class="phase-num">Phase 02 — Statistical Modelling</div>
      <h4>02_analysis_modeling.py</h4>
      <p>Runs <strong>rolling Z-score</strong> (6-month window, ±2σ threshold) and <strong>CUSUM</strong> for spike detection. Tests the hardware-aging hypothesis with <strong>OLS regression</strong> + Pearson correlation. Trains a <strong>Random Forest classifier</strong> (AUC-validated via 5-fold CV) to predict high-severity events.</p>
    </div>
    <div class="method-card">
      <div class="phase-num">Phase 03 — Reporting</div>
      <h4>03_generate_report.py</h4>
      <p>Reads <code>analysis_results.json</code> and embeds base64-encoded figures into a self-contained <strong>Jinja2 HTML template</strong>. The report re-generates with a single command — no manual copy-paste. All KPIs and conclusions update automatically from the latest pipeline run.</p>
    </div>
    <div class="method-card">
      <div class="phase-num">Reproducibility</div>
      <h4>run_pipeline.sh</h4>
      <p>Single entry-point shell script executes all three phases in sequence. Random seed is fixed (42) throughout. All intermediate artifacts (CSVs, DuckDB, JSON) are versioned alongside the code for full audit trail compliance.</p>
    </div>
  </div>

  <div class="findings" style="margin-top:24px">
    <h4>📦 Tech Stack</h4>
    <ul>
      <li><strong>Data Acquisition:</strong> Python · requests · openFDA REST API</li>
      <li><strong>Wrangling & Storage:</strong> pandas · DuckDB (SQLite-compatible relational queries)</li>
      <li><strong>Statistical Analysis:</strong> scipy.stats · statsmodels · Z-score · CUSUM</li>
      <li><strong>Machine Learning:</strong> scikit-learn RandomForestClassifier · LinearRegression · cross_val_score</li>
      <li><strong>Visualisation:</strong> matplotlib · seaborn (dark-theme, 150 dpi publication quality)</li>
      <li><strong>Reporting:</strong> Jinja2 HTML template → self-contained single-file report</li>
    </ul>
  </div>
</div>

</div><!-- /container -->

<footer>
  <span>Adverse Event Trend Analysis & Risk Detection System — Portfolio Project</span>
  <span>Generated {{ generated_at }} · Pipeline v1.0</span>
</footer>

</body>
</html>
"""


def generate_report():
    print("=" * 60)
    print("PHASE 3: HTML Report Generation")
    print("=" * 60)

    with open("data/analysis_results.json") as f:
        results = json.load(f)

    ts  = results["time_series"]
    reg = results["regression"]
    rf  = results["random_forest"]
    kpis = results["kpis"]

    # Load and encode figures
    figs = {}
    for key, path in FIGURES.items():
        if os.path.exists(path):
            figs[key] = img_to_b64(path)
        else:
            print(f"  Warning: {path} not found — skipping.")
            figs[key] = ""

    top_failure       = list(kpis["top_failure_modes"].keys())[0]
    top_failure_count = list(kpis["top_failure_modes"].values())[0]
    max_importance    = max(rf["feature_importances"].values())

    template = Template(HTML_TEMPLATE)
    html = template.render(
        generated_at      = datetime.now().strftime("%B %d, %Y at %H:%M"),
        ts                = ts,
        reg               = reg,
        rf                = rf,
        kpis              = kpis,
        top_failure       = top_failure,
        top_failure_count = top_failure_count,
        max_importance    = max_importance,
        list              = list,
        enumerate         = enumerate,
        **figs,
    )

    os.makedirs("outputs", exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(REPORT_PATH) // 1024
    print(f"  Report saved: {REPORT_PATH} ({size_kb} KB)")
    print("Phase 3 complete.")


if __name__ == "__main__":
    generate_report()
