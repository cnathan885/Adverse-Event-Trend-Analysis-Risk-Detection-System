"""
Phase 2: Analysis & Statistical Modeling
==========================================
- Time-series trend analysis with anomaly / spike detection (Z-score + CUSUM)
- Device aging vs. failure severity regression
- Random Forest classifier for high-risk event prediction
- Outputs plots + model metrics for the Phase 3 report

Author: Portfolio Project — Adverse Event Trend Analysis & Risk Detection System
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
import warnings
import json
import os

warnings.filterwarnings("ignore")

# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE = {
    "bg":       "#0D1117",
    "panel":    "#161B22",
    "border":   "#30363D",
    "accent1":  "#58A6FF",   # blue
    "accent2":  "#F78166",   # red-orange
    "accent3":  "#3FB950",   # green
    "accent4":  "#D2A8FF",   # purple
    "text":     "#E6EDF3",
    "subtext":  "#8B949E",
}

sns.set_theme(style="dark", rc={
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["panel"],
    "axes.edgecolor":    PALETTE["border"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["subtext"],
    "ytick.color":       PALETTE["subtext"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["border"],
    "grid.linewidth":    0.6,
    "legend.facecolor":  PALETTE["panel"],
    "legend.edgecolor":  PALETTE["border"],
})

os.makedirs("outputs", exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    events  = pd.read_csv("data/cleaned_events.csv", parse_dates=["date_of_event","date_received"])
    devices = pd.read_csv("data/device_master.csv")
    print(f"Loaded {len(events)} events and {len(devices)} device master records.")
    return events, devices


# ── 2A: Time-Series Trend Analysis ────────────────────────────────────────────

def run_time_series_analysis(df: pd.DataFrame) -> dict:
    """
    Monthly event counts per device.
    Spike detection: rolling Z-score (> 2σ) and CUSUM algorithm.
    Returns spike records and saves a chart.
    """
    print("\n── 2A: Time-Series Trend Analysis ───────────────────────────────")

    monthly = (
        df.groupby([df["date_of_event"].dt.to_period("M"), "device_name"])
          .size()
          .reset_index(name="event_count")
    )
    monthly["date"] = monthly["date_of_event"].dt.to_timestamp()

    # Aggregate total across all devices for a macro view
    total_monthly = (
        monthly.groupby("date")["event_count"].sum().reset_index()
    )
    total_monthly = total_monthly.sort_values("date")

    # Rolling mean + std (6-month window)
    window = 6
    total_monthly["rolling_mean"] = total_monthly["event_count"].rolling(window, min_periods=2).mean()
    total_monthly["rolling_std"]  = total_monthly["event_count"].rolling(window, min_periods=2).std()
    total_monthly["z_score"]      = (
        (total_monthly["event_count"] - total_monthly["rolling_mean"])
        / total_monthly["rolling_std"].replace(0, np.nan)
    )
    total_monthly["spike_zscore"] = total_monthly["z_score"].abs() > 2.0

    # CUSUM: cumulative sum of deviations from target (mean)
    target = total_monthly["event_count"].mean()
    slack  = 0.5 * total_monthly["event_count"].std()
    cusum_pos, cusum_neg = [0.0], [0.0]
    for val in total_monthly["event_count"].iloc[1:]:
        cusum_pos.append(max(0, cusum_pos[-1] + val - (target + slack)))
        cusum_neg.append(min(0, cusum_neg[-1] + val - (target - slack)))
    total_monthly["cusum_pos"] = cusum_pos
    control_limit = 4 * total_monthly["event_count"].std()
    total_monthly["cusum_alarm"] = total_monthly["cusum_pos"] > control_limit

    spikes = total_monthly[(total_monthly["spike_zscore"]) | (total_monthly["cusum_alarm"])].copy()
    print(f"  Detected {len(spikes)} spike months via Z-score or CUSUM")

    # Per-device monthly for heatmap
    pivot = monthly.pivot_table(
        index="device_name", columns="date", values="event_count", aggfunc="sum", fill_value=0
    )

    # ── Figure 1: Trend chart with spike overlays ──────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios":[3,1]})
    fig.patch.set_facecolor(PALETTE["bg"])

    ax1 = axes[0]
    ax1.plot(total_monthly["date"], total_monthly["event_count"],
             color=PALETTE["accent1"], lw=1.8, label="Monthly Events", zorder=3)
    ax1.fill_between(total_monthly["date"], total_monthly["event_count"],
                     alpha=0.12, color=PALETTE["accent1"])
    ax1.plot(total_monthly["date"], total_monthly["rolling_mean"],
             color=PALETTE["accent3"], lw=1.4, ls="--", label=f"{window}-mo Rolling Mean")

    upper = total_monthly["rolling_mean"] + 2 * total_monthly["rolling_std"]
    lower = total_monthly["rolling_mean"] - 2 * total_monthly["rolling_std"]
    ax1.fill_between(total_monthly["date"], lower.clip(0), upper,
                     alpha=0.10, color=PALETTE["accent3"], label="±2σ Band")

    # Highlight spikes
    for _, row in spikes.iterrows():
        ax1.axvline(row["date"], color=PALETTE["accent2"], lw=1.0, alpha=0.6, ls=":")
        ax1.scatter(row["date"], row["event_count"],
                    color=PALETTE["accent2"], s=80, zorder=5)

    ax1.set_title("Adverse Event Monthly Volume with Spike Detection",
                  fontsize=14, color=PALETTE["text"], pad=12)
    ax1.set_ylabel("Event Count", color=PALETTE["text"])
    ax1.legend(loc="upper left", fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # CUSUM subplot
    ax2 = axes[1]
    ax2.plot(total_monthly["date"], total_monthly["cusum_pos"],
             color=PALETTE["accent4"], lw=1.5, label="CUSUM+")
    ax2.axhline(control_limit, color=PALETTE["accent2"], lw=1.2, ls="--",
                label=f"Control Limit ({control_limit:.0f})")
    ax2.fill_between(total_monthly["date"], 0, total_monthly["cusum_pos"],
                     where=total_monthly["cusum_alarm"],
                     color=PALETTE["accent2"], alpha=0.25, label="CUSUM Alarm")
    ax2.set_ylabel("CUSUM+", color=PALETTE["text"])
    ax2.set_xlabel("Date", color=PALETTE["text"])
    ax2.legend(loc="upper left", fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    fig.savefig("outputs/fig1_trend_spike_detection.png", dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print("  Saved fig1_trend_spike_detection.png")

    # ── Figure 2: Device-level heatmap ────────────────────────────────────
    # Sample columns to keep heatmap readable
    col_sample = pivot.columns[::3]
    pivot_sample = pivot[col_sample]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(PALETTE["bg"])
    sns.heatmap(
        pivot_sample,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.3,
        linecolor=PALETTE["bg"],
        cbar_kws={"label": "Event Count", "shrink": 0.6},
        fmt=".0f",
        annot=pivot_sample > 10,
    )
    ax.set_title("Event Frequency Heatmap by Device & Month (every 3rd month shown)",
                 fontsize=12, color=PALETTE["text"], pad=10)
    ax.set_xlabel("Month", color=PALETTE["text"])
    ax.set_ylabel("Device Type", color=PALETTE["text"])
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    plt.tight_layout()
    fig.savefig("outputs/fig2_device_heatmap.png", dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print("  Saved fig2_device_heatmap.png")

    return {
        "spike_months": len(spikes),
        "total_months":  len(total_monthly),
        "spike_dates":   spikes["date"].dt.strftime("%Y-%m").tolist(),
        "monthly_data":  total_monthly[["date","event_count","rolling_mean","z_score"]].to_dict("records"),
    }


# ── 2B: Aging vs. Failure Severity Regression ─────────────────────────────────

def run_aging_regression(df: pd.DataFrame) -> dict:
    """
    Linear regression: device_age_years → severity_score
    Confirms or refutes the hardware-aging hypothesis.
    """
    print("\n── 2B: Hardware Aging vs. Severity Regression ───────────────────")

    reg_df = df.dropna(subset=["device_age_years", "severity_score"]).copy()
    X = reg_df[["device_age_years"]].values
    y = reg_df["severity_score"].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    r2   = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    corr, pval = stats.pearsonr(reg_df["device_age_years"], reg_df["severity_score"])

    print(f"  R²={r2:.4f}  RMSE={rmse:.4f}  Pearson r={corr:.4f}  p={pval:.4e}")

    # Per-device regression lines
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax = axes[0]
    ax.scatter(reg_df["device_age_years"], reg_df["severity_score"],
               alpha=0.15, s=18, color=PALETTE["accent1"], label="Events")
    ax.plot(np.linspace(0, reg_df["device_age_years"].max(), 200),
            model.predict(np.linspace(0, reg_df["device_age_years"].max(), 200).reshape(-1,1)),
            color=PALETTE["accent2"], lw=2, label=f"OLS fit (R²={r2:.3f})")
    ax.set_xlabel("Device Age (years)")
    ax.set_ylabel("Severity Score (0–5)")
    ax.set_title("Device Age vs. Patient Outcome Severity")
    ax.legend(fontsize=9)

    # Box plot: age buckets
    ax2 = axes[1]
    reg_df["age_bucket"] = pd.cut(reg_df["device_age_years"],
                                  bins=[0,2,4,6,8,30],
                                  labels=["<2yr","2–4yr","4–6yr","6–8yr","8yr+"])
    bp_data = [
        reg_df.loc[reg_df["age_bucket"]==b, "severity_score"].dropna().values
        for b in ["<2yr","2–4yr","4–6yr","6–8yr","8yr+"]
    ]
    bp = ax2.boxplot(bp_data, labels=["<2yr","2–4yr","4–6yr","6–8yr","8yr+"],
                     patch_artist=True, medianprops=dict(color=PALETTE["accent3"], lw=2))
    colors = [PALETTE["accent1"], PALETTE["accent1"], PALETTE["accent4"],
              PALETTE["accent4"], PALETTE["accent2"]]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_xlabel("Device Age Bucket")
    ax2.set_ylabel("Severity Score")
    ax2.set_title("Severity Distribution by Age Group")

    plt.tight_layout()
    fig.savefig("outputs/fig3_aging_regression.png", dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print("  Saved fig3_aging_regression.png")

    return {
        "r2": round(r2, 4), "rmse": round(rmse, 4),
        "pearson_r": round(corr, 4), "pearson_p": float(f"{pval:.4e}"),
        "coefficient": round(float(model.coef_[0]), 4),
        "intercept":   round(float(model.intercept_), 4),
        "n": len(reg_df),
    }


# ── 2C: Random Forest Risk Classifier ─────────────────────────────────────────

def run_risk_classifier(df: pd.DataFrame) -> dict:
    """
    Random Forest to predict high-severity events (severity ≥ 3).
    Features: device type, age, failure mode, days_to_report, month.
    Reports feature importance and AUC.
    """
    print("\n── 2C: Random Forest Risk Classifier ────────────────────────────")

    ml_df = df.dropna(subset=["device_age_years","severity_score","failure_mode_clean"]).copy()
    ml_df["high_risk"] = (ml_df["severity_score"] >= 3).astype(int)

    le_device  = LabelEncoder()
    le_failure = LabelEncoder()
    ml_df["device_enc"]  = le_device.fit_transform(ml_df["device_name"].fillna("Unknown"))
    ml_df["failure_enc"] = le_failure.fit_transform(ml_df["failure_mode_clean"].fillna("other"))

    feature_cols = ["device_enc", "device_age_years", "failure_enc",
                    "days_to_report", "month", "late_report"]
    X = ml_df[feature_cols].fillna(0).values
    y = ml_df["high_risk"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(X_train, y_train)
    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    auc     = roc_auc_score(y_test, y_proba)
    cv_aucs = cross_val_score(rf, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
    report  = classification_report(y_test, y_pred, output_dict=True)

    feature_names = ["Device Type", "Device Age (yrs)", "Failure Mode",
                     "Days to Report", "Month of Year", "Late Report Flag"]
    importances   = rf.feature_importances_
    fi_sorted     = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print(f"  AUC={auc:.4f}  CV-AUC={cv_aucs.mean():.4f}±{cv_aucs.std():.4f}")
    print(f"  High-risk precision: {report['1']['precision']:.3f}  recall: {report['1']['recall']:.3f}")

    # ── Figure 4: Feature importance + confusion matrix ───────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax1 = axes[0]
    names  = [x[0] for x in fi_sorted]
    values = [x[1] for x in fi_sorted]
    bars   = ax1.barh(names[::-1], values[::-1], color=PALETTE["accent1"], alpha=0.85, height=0.6)
    for bar, val in zip(bars, values[::-1]):
        ax1.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", color=PALETTE["text"], fontsize=9)
    ax1.set_xlabel("Feature Importance (Gini)")
    ax1.set_title(f"RF Feature Importances  |  AUC={auc:.3f}")
    ax1.set_xlim(0, max(values)*1.18)

    ax2 = axes[1]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2,
                xticklabels=["Low Risk","High Risk"],
                yticklabels=["Low Risk","High Risk"],
                linewidths=1, linecolor=PALETTE["bg"],
                cbar_kws={"shrink":0.7})
    ax2.set_title("Confusion Matrix — Test Set")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")

    plt.tight_layout()
    fig.savefig("outputs/fig4_rf_results.png", dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print("  Saved fig4_rf_results.png")

    return {
        "auc": round(auc, 4),
        "cv_auc_mean": round(float(cv_aucs.mean()), 4),
        "cv_auc_std":  round(float(cv_aucs.std()),  4),
        "precision_high_risk": round(report["1"]["precision"], 4),
        "recall_high_risk":    round(report["1"]["recall"],    4),
        "f1_high_risk":        round(report["1"]["f1-score"],  4),
        "feature_importances": {n: round(v, 4) for n, v in fi_sorted},
        "n_train": len(X_train),
        "n_test":  len(X_test),
    }


# ── 2D: Failure Mode & KPI Summary ────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame, devices_df: pd.DataFrame) -> dict:
    """Compute KPI table for the Phase 3 report."""
    print("\n── 2D: KPI Summary Computation ──────────────────────────────────")

    total       = len(df)
    high_risk   = int((df["severity_score"] >= 3).sum())
    deaths      = int((df["patient_outcome"] == "death").sum())
    late        = int(df["late_report"].sum())
    avg_days    = round(float(df["days_to_report"].mean()), 1)
    compliance  = round((1 - late / total) * 100, 1)
    avg_age     = round(float(df["device_age_years"].mean()), 2)
    avg_sev     = round(float(df["severity_score"].dropna().mean()), 3)

    # Top failure modes
    top_failures = (
        df["failure_mode_clean"]
          .value_counts()
          .head(5)
          .to_dict()
    )

    # Device risk ranking
    device_risk = (
        df.groupby("device_name")["severity_score"]
          .agg(["mean","count"])
          .rename(columns={"mean":"avg_severity","count":"n_events"})
          .sort_values("avg_severity", ascending=False)
          .round(3)
          .to_dict("index")
    )

    # Reporting compliance by manufacturer
    mfr_compliance = (
        df.groupby("manufacturer")
          .agg(total_reports=("report_id","count"),
               late_reports=("late_report","sum"))
          .assign(compliance_pct=lambda x: (1 - x["late_reports"]/x["total_reports"]).round(3)*100)
          .sort_values("compliance_pct")
          .head(10)
          .to_dict("index")
    )

    # ── Figure 5: KPI & Failure Mode Breakdown ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("Product Quality KPI Dashboard", fontsize=14, color=PALETTE["text"],
                 y=1.01, fontweight="bold")

    # Panel 1: Outcome distribution
    outcome_counts = df["patient_outcome"].value_counts()
    wedge_colors   = [PALETTE["accent3"], PALETTE["accent1"], PALETTE["accent4"],
                      "#E3B341", PALETTE["accent2"], "#FF6E6E", PALETTE["subtext"]]
    axes[0].pie(outcome_counts.values,
                labels=outcome_counts.index,
                colors=wedge_colors[:len(outcome_counts)],
                autopct="%1.1f%%", startangle=140,
                textprops={"color": PALETTE["text"], "fontsize": 8},
                wedgeprops={"edgecolor": PALETTE["bg"], "linewidth": 1.5})
    axes[0].set_title("Patient Outcome Distribution", color=PALETTE["text"])

    # Panel 2: Top failure modes
    modes   = list(top_failures.keys())
    counts  = list(top_failures.values())
    colors2 = [PALETTE["accent1"], PALETTE["accent4"], PALETTE["accent3"],
               "#E3B341", PALETTE["accent2"]]
    axes[1].barh(modes[::-1], counts[::-1], color=colors2, alpha=0.85, height=0.6)
    axes[1].set_title("Top 5 Failure Modes", color=PALETTE["text"])
    axes[1].set_xlabel("Frequency")

    # Panel 3: Reporting compliance by device
    dev_comp = (
        df.groupby("device_name")
          .apply(lambda x: (1 - x["late_report"].sum()/len(x))*100)
          .sort_values()
    )
    clr = [PALETTE["accent2"] if v < 75 else PALETTE["accent1"] if v < 90 else PALETTE["accent3"]
           for v in dev_comp.values]
    axes[2].barh(dev_comp.index, dev_comp.values, color=clr, alpha=0.85, height=0.6)
    axes[2].axvline(80, color=PALETTE["accent2"], lw=1.2, ls="--", label="80% threshold")
    axes[2].set_xlim(0, 105)
    axes[2].set_title("Reporting Compliance % by Device", color=PALETTE["text"])
    axes[2].set_xlabel("Compliance (%)")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig("outputs/fig5_kpi_dashboard.png", dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print("  Saved fig5_kpi_dashboard.png")

    return {
        "total_reports":       total,
        "high_risk_events":    high_risk,
        "death_events":        deaths,
        "late_reports":        late,
        "avg_days_to_report":  avg_days,
        "compliance_pct":      compliance,
        "avg_device_age_yrs":  avg_age,
        "avg_severity_score":  avg_sev,
        "top_failure_modes":   top_failures,
        "device_risk_ranking": device_risk,
        "mfr_compliance":      mfr_compliance,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 2: Analysis & Statistical Modeling")
    print("=" * 60)

    events_df, devices_df = load_data()

    ts_results  = run_time_series_analysis(events_df)
    reg_results = run_aging_regression(events_df)
    rf_results  = run_risk_classifier(events_df)
    kpi_results = compute_kpis(events_df, devices_df)

    all_results = {
        "time_series":   ts_results,
        "regression":    reg_results,
        "random_forest": rf_results,
        "kpis":          kpi_results,
    }

    # Serialise dates for JSON
    def default(o):
        if hasattr(o, "isoformat"): return o.isoformat()
        if isinstance(o, (np.int64, np.int32)): return int(o)
        if isinstance(o, (np.float64, np.float32)): return float(o)
        return str(o)

    with open("data/analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=default)

    print("\nPhase 2 complete. Results saved to data/analysis_results.json\n")


if __name__ == "__main__":
    main()
