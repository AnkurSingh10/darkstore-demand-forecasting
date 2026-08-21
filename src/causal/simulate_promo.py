import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np

file_path = Path(__file__).resolve()
root_dir = file_path.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    from src.data_loader import load_and_process_demand
except ModuleNotFoundError:
    from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    from data_loader import load_and_process_demand

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_panel_units(
    df: pd.DataFrame, regions: List[str] = ["region_north", "region_south", "region_east"]
) -> pd.DataFrame:
    """
    Expands department daily demand into synthetic department-region units.

    Args:
        df (pd.DataFrame): Daily demand DataFrame with 'date', 'department', and 'demand'.
        regions (List[str]): List of region names.

    Returns:
        pd.DataFrame: Panel dataset with 'unit_id', 'department', 'region', 'date', 'demand'.
    """
    records = []
    np.random.seed(42)

    for region in regions:
        # Scale demand slightly per region 
        region_factor = np.random.uniform(0.8, 1.2)
        df_region = df.copy()
        df_region["region"] = region
        df_region["unit_id"] = df_region["department"].astype(str) + "_" + region
        
        # Adding slight regional variance to baseline demand
        noise = np.random.normal(1.0, 0.05, size=len(df_region))
        df_region["demand"] = (df_region["demand"] * region_factor * noise).clip(lower=1).astype(int)
        records.append(df_region)

    panel_df = pd.concat(records, ignore_index=True)
    panel_df["date"] = pd.to_datetime(panel_df["date"])
    panel_df = panel_df.sort_values(by=["unit_id", "date"]).reset_index(drop=True)
    logger.info(f"Created panel dataset with {panel_df['unit_id'].nunique()} unique units across {len(regions)} regions.")
    return panel_df


def simulate_promo_treatment(
    panel_df: pd.DataFrame,
    treatment_start_date: str = "2024-09-01",
    treatment_ratio: float = 0.5,
    true_uplift_pct: float = 0.20,
    noise_std: float = 0.03,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Simulates promotional campaign treatment on a subset of units starting from a specified date.

    Args:
        panel_df (pd.DataFrame): Panel DataFrame.
        treatment_start_date (str): Cutoff date for promo launch.
        treatment_ratio (float): Fraction of units selected for treatment.
        true_uplift_pct (float): Ground truth percentage demand increase (e.g. 0.20 = 20%).
        noise_std (float): Standard deviation of random noise added to treatment effect.
        random_seed (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, Dict]: (Labeled panel DataFrame, Ground truth metadata dictionary).
    """
    np.random.seed(random_seed)
    df = panel_df.copy()
    treatment_date = pd.to_datetime(treatment_start_date)

    # Randomly select treated units
    unique_units = np.sort(df["unit_id"].unique())
    num_treated = int(len(unique_units) * treatment_ratio)
    treated_units = set(np.random.choice(unique_units, size=num_treated, replace=False))

    logger.info(
        f"Selected {len(treated_units)} treated units out of {len(unique_units)} total units "
        f"starting from treatment date {treatment_start_date}."
    )

    # Adding treatment flags
    df["treated"] = df["unit_id"].apply(lambda u: 1 if u in treated_units else 0)
    df["post_treatment"] = (df["date"] >= treatment_date).astype(int)

    # Store pre-treatment ground truth demand
    df["demand_untreated"] = df["demand"].copy()

    # Inject synthetic demand uplift for treated units in post-treatment period
    mask = (df["treated"] == 1) & (df["post_treatment"] == 1)
    n_treated_obs = mask.sum()

    if n_treated_obs > 0:
        # Multiplicative uplift: 1 + true_uplift + Gaussian noise
        uplift_multipliers = 1.0 + true_uplift_pct + np.random.normal(0, noise_std, size=n_treated_obs)
        uplift_multipliers = np.clip(uplift_multipliers, 1.0, 2.0)
        
        df.loc[mask, "demand"] = np.round(df.loc[mask, "demand"] * uplift_multipliers).astype(int)

    # Required output columns
    output_cols = ["unit_id", "date", "demand", "treated", "post_treatment"]
    result_df = df[output_cols].copy()

    # Ground truth metadata
    ground_truth_meta = {
        "treatment_start_date": treatment_start_date,
        "true_uplift_pct": true_uplift_pct,
        "treatment_ratio": treatment_ratio,
        "num_total_units": int(len(unique_units)),
        "num_treated_units": int(len(treated_units)),
        "treated_units": sorted(list(treated_units)),
        "control_units": sorted(list(set(unique_units) - treated_units)),
    }

    return result_df, ground_truth_meta


def run_simulation(
    input_parquet: Path = PROCESSED_DATA_DIR / "daily_demand.parquet",
    output_parquet: Path = PROCESSED_DATA_DIR / "promo_simulated_demand.parquet",
    ground_truth_json: Path = PROCESSED_DATA_DIR / "promo_ground_truth.json",
    treatment_start_date: str = "2024-09-01",
    true_uplift_pct: float = 0.20,
) -> pd.DataFrame:
    """
    Runs end-to-end promo simulation pipeline and exports labeled dataset + metadata.

    Args:
        input_parquet (Path): Path to daily_demand.parquet.
        output_parquet (Path): Path to output promo_simulated_demand.parquet.
        ground_truth_json (Path): Metadata JSON path.
        treatment_start_date (str): Treatment launch date.
        true_uplift_pct (float): Ground truth demand uplift percentage.

    Returns:
        pd.DataFrame: Simulated labeled dataset.
    """
    if not input_parquet.exists():
        logger.info(f"Input dataset missing at {input_parquet}. Running data loader pipeline...")
        raw_df = load_and_process_demand(data_dir=RAW_DATA_DIR, output_file=input_parquet)
    else:
        raw_df = pd.read_parquet(input_parquet)

    panel_df = create_panel_units(raw_df)
    simulated_df, meta = simulate_promo_treatment(
        panel_df,
        treatment_start_date=treatment_start_date,
        true_uplift_pct=true_uplift_pct,
    )

    # Save output parquet
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    simulated_df.to_parquet(output_parquet, index=False)
    logger.info(f"Saved simulated promo dataset to '{output_parquet}' ({len(simulated_df):,} rows).")

    # Save ground truth metadata JSON
    with open(ground_truth_json, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved ground-truth experiment metadata to '{ground_truth_json}'.")

    return simulated_df


if __name__ == "__main__":
    df_sim = run_simulation()
    print("\nSimulated Promo Dataset Preview:")
    print(df_sim.head(10))
    print("\nTreated vs Control Unit Counts:")
    print(df_sim.groupby("treated")["unit_id"].nunique())