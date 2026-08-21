import sys
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

file_path = Path(__file__).resolve()
root_dir = file_path.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    from src.causal.simulate_promo import run_simulation
except ModuleNotFoundError:
    from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    from simulate_promo import run_simulation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_causal_data(
    data_parquet: Path = PROCESSED_DATA_DIR / "promo_simulated_demand.parquet",
    meta_json: Path = PROCESSED_DATA_DIR / "promo_ground_truth.json",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Loads simulated promo panel data and ground-truth metadata.

    Args:
        data_parquet (Path): Path to promo_simulated_demand.parquet.
        meta_json (Path): Path to promo_ground_truth.json.

    Returns:
        Tuple[pd.DataFrame, Dict]: (Panel DataFrame, Metadata dictionary)
    """
    if not data_parquet.exists() or not meta_json.exists():
        logger.info("Simulated dataset or metadata missing. Running promo simulation first...")
        df = run_simulation(output_parquet=data_parquet, ground_truth_json=meta_json)
        with open(meta_json, "r") as f:
            meta = json.load(f)
    else:
        df = pd.read_parquet(data_parquet)
        with open(meta_json, "r") as f:
            meta = json.load(f)

    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"Loaded simulated dataset with {len(df):,} rows and metadata for {meta['num_treated_units']} treated units.")
    return df, meta


def fit_did_regression(df: pd.DataFrame) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, Dict[str, float]]:
    """
    Fits Difference-in-Differences OLS regression model:
    demand ~ treated + post_treatment + treated:post_treatment

    Args:
        df (pd.DataFrame): Labeled panel dataset with 'demand', 'treated', 'post_treatment'.

    Returns:
        Tuple: (Fitted OLS model object, Result metrics dict)
    """
    logger.info("Fitting Difference-in-Differences regression model...")
    
    # Fitting OLS model with robust standard errors clustered by unit_id
    formula = "demand ~ treated + post_treatment + treated:post_treatment"
    model = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["unit_id"]})

    # Interaction term coefficient is the treatment effect (ATT)
    interaction_col = "treated:post_treatment"
    att_beta = model.params[interaction_col]
    att_se = model.bse[interaction_col]
    att_pval = model.pvalues[interaction_col]
    ci_lower, ci_upper = model.conf_int().loc[interaction_col]

    # Calculate baseline demand for treated group in pre-treatment period
    pre_treated_mask = (df["treated"] == 1) & (df["post_treatment"] == 0)
    baseline_treated_demand = df.loc[pre_treated_mask, "demand"].mean()

    # Percentage uplift estimated by DiD
    est_uplift_pct = (att_beta / baseline_treated_demand) * 100.0

    metrics = {
        "att_beta": round(float(att_beta), 2),
        "att_se": round(float(att_se), 2),
        "p_value": round(float(att_pval), 5),
        "ci_lower": round(float(ci_lower), 2),
        "ci_upper": round(float(ci_upper), 2),
        "baseline_demand": round(float(baseline_treated_demand), 2),
        "estimated_uplift_pct": round(float(est_uplift_pct), 2),
    }

    logger.info(f"DiD Regression Fit Complete. Estimated ATT Beta = {att_beta:.2f} (p={att_pval:.5f}).")
    return model, metrics


def plot_parallel_trends(
    df: pd.DataFrame,
    treatment_start_date: str,
    output_plot_path: Path = Path("reports/diff_in_diff_parallel_trends.png"),
):
    """
    Plots daily mean demand trajectory for Treated vs Control units over time to visually check
    the parallel trends assumption prior to treatment date.

    Args:
        df (pd.DataFrame): Panel DataFrame.
        treatment_start_date (str): Cutoff date for promo launch.
        output_plot_path (Path): Chart export destination.
    """
    logger.info("Generating Parallel Trends validation plot...")

    # Group daily mean demand by date and treatment status
    trend_df = (
        df.groupby(["date", "treated"])["demand"]
        .mean()
        .reset_index()
        .sort_values("date")
    )

    plt.figure(figsize=(13, 6))
    
    control_trend = trend_df[trend_df["treated"] == 0]
    treated_trend = trend_df[trend_df["treated"] == 1]

    plt.plot(control_trend["date"], control_trend["demand"], label="Control Group (No Promo)", color="#1f77b4", linewidth=2.0)
    plt.plot(treated_trend["date"], treated_trend["demand"], label="Treated Group (Promo)", color="#d95f02", linewidth=2.0)

    # Vertical line indicating treatment start date
    t_date = pd.to_datetime(treatment_start_date)
    plt.axvline(t_date, color="red", linestyle="--", linewidth=2.0, label=f"Promo Start ({treatment_start_date})")

    # Annotate Pre-treatment and Post-treatment windows
    plt.title("Difference-in-Differences: Parallel Trends Check & Treatment Impact", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Mean Daily Demand per Unit", fontsize=12)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(rotation=30)
    plt.tight_layout()

    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=300)
    logger.info(f"Saved parallel trends plot to '{output_plot_path}'.")
    plt.close()



def run_did_analysis(
    data_parquet: Path = PROCESSED_DATA_DIR / "promo_simulated_demand.parquet",
    meta_json: Path = PROCESSED_DATA_DIR / "promo_ground_truth.json",
    output_plot_path: Path = Path("reports/diff_in_diff_parallel_trends.png"),
) -> Dict[str, float]:
    """
    Runs end-to-end Difference-in-Differences causal inference pipeline.

    Args:
        data_parquet (Path): Path to promo_simulated_demand.parquet.
        meta_json (Path): Path to ground-truth metadata.
        output_plot_path (Path): Path for parallel trends visualization.

    Returns:
        Dict[str, float]: Summary dictionary of estimated vs true causal effects.
    """
    df, meta = load_causal_data(data_parquet, meta_json)
    model, metrics = fit_did_regression(df)

    # Calculate true injected uplift percentage
    true_uplift_pct = meta["true_uplift_pct"] * 100.0
    est_uplift_pct = metrics["estimated_uplift_pct"]
    diff_pct = est_uplift_pct - true_uplift_pct

    # Plot Parallel Trends
    plot_parallel_trends(df, meta["treatment_start_date"], output_plot_path=output_plot_path)

    # Print Plain-English Summary
    summary_text = (
        f"  DIFFERENCE-IN-DIFFERENCES CAUSAL EVALUATION   \n"
        f"Estimated uplift: {est_uplift_pct:.2f}%, True uplift: {true_uplift_pct:.2f}%, Difference: {diff_pct:+.2f}%\n"
        f"Absolute Unit Demand Increase (ATT Beta): +{metrics['att_beta']} items/day\n"
        f"95% Confidence Interval: [{metrics['ci_lower']}, {metrics['ci_upper']}]\n"
        f"Statistical Significance: p-value = {metrics['p_value']} "
        f"({'Statistically Significant (p < 0.05)' if metrics['p_value'] < 0.05 else 'Not Significant'})\n"
    )
    print(summary_text)

    return {
        "estimated_uplift_pct": est_uplift_pct,
        "true_uplift_pct": true_uplift_pct,
        "difference_pct": round(diff_pct, 2),
        "p_value": metrics["p_value"],
    }


if __name__ == "__main__":
    run_did_analysis()
