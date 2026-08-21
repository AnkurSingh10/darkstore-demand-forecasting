import sys
import logging
from pathlib import Path
from typing import List, Optional

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

# List of major Indian holidays (proxy for promotional peak demand windows)
INDIAN_HOLIDAYS = [
    "2023-01-01", "2023-01-26", "2023-03-08", "2023-08-15", "2023-08-30", "2023-10-02", "2023-10-24", "2023-11-12", "2023-12-25",
    "2024-01-01", "2024-01-26", "2024-03-25", "2024-08-15", "2024-08-19", "2024-10-02", "2024-10-12", "2024-11-01", "2024-12-25",
    "2025-01-01", "2025-01-26", "2025-03-14", "2025-08-15", "2025-10-02", "2025-10-20", "2025-12-25"
]


def add_lag_features(df: pd.DataFrame, lags: List[int] = [1, 7, 14, 28], group_col: str = "department") -> pd.DataFrame:
    """
    Creates historical demand lag features per category group.

    Args:
        df (pd.DataFrame): Time series DataFrame sorted by group and date.
        lags (List[int]): List of lag days to compute.
        group_col (str): Category grouping column name.

    Returns:
        pd.DataFrame: DataFrame with added lag columns.
    """
    logger.info(f"Computing lag features for lags {lags}...")
    for lag in lags:
        df[f"demand_lag_{lag}"] = df.groupby(group_col, observed=False)["demand"].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, windows: List[int] = [7, 28], group_col: str = "department"
) -> pd.DataFrame:
    """
    Computes rolling mean and standard deviation features per category group.
    Shifts demand by 1 day before rolling to prevent data leakage of the target variable.

    Args:
        df (pd.DataFrame): Time series DataFrame sorted by group and date.
        windows (List[int]): Rolling window lengths (in days).
        group_col (str): Category grouping column name.

    Returns:
        pd.DataFrame: DataFrame with added rolling mean and std columns.
    """
    logger.info(f"Computing rolling statistics for window sizes {windows}...")
    for window in windows:
        # Shift by 1 to prevent data leakage of current day demand
        df[f"demand_rolling_mean_{window}"] = (
            df.groupby(group_col, observed=False)["demand"]
            .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
        )
        df[f"demand_rolling_std_{window}"] = (
            df.groupby(group_col, observed=False)["demand"]
            .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).std())
            .fillna(0)
        )
    return df



def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts calendar signals including day of week, weekend indicator, and month.

    Args:
        df (pd.DataFrame): DataFrame containing a 'date' column.

    Returns:
        pd.DataFrame: DataFrame with calendar features.
    """
    logger.info("Computing calendar features (day_of_week, is_weekend, month)...")
    date_series = pd.to_datetime(df["date"])
    df["day_of_week"] = date_series.dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = date_series.dt.month
    return df


def add_holiday_features(df: pd.DataFrame, holidays: List[str] = INDIAN_HOLIDAYS) -> pd.DataFrame:
    """
    Computes days elapsed since the most recent major holiday/festival.

    Args:
        df (pd.DataFrame): DataFrame with 'date' column.
        holidays (List[str]): List of holiday YYYY-MM-DD date strings.

    Returns:
        pd.DataFrame: DataFrame with 'days_since_last_holiday' column.
    """
    logger.info("Computing days_since_last_holiday feature...")
    
    unique_dates = pd.DataFrame({"date": pd.to_datetime(df["date"].unique())}).sort_values("date")
    holidays_df = pd.DataFrame({"date": pd.to_datetime(holidays), "holiday_date": pd.to_datetime(holidays)}).sort_values("date")
    
    # Merge asof backward to find most recent prior holiday
    merged_dates = pd.merge_asof(unique_dates, holidays_df, on="date", direction="backward")
    merged_dates["days_since_last_holiday"] = (
        (merged_dates["date"] - merged_dates["holiday_date"]).dt.days.fillna(999).astype(int)
    )
    
    # Convert date back to original format string or Timestamp for merging
    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        date_map = dict(zip(merged_dates["date"], merged_dates["days_since_last_holiday"]))
    else:
        date_map = dict(zip(merged_dates["date"].dt.strftime("%Y-%m-%d"), merged_dates["days_since_last_holiday"]))
        
    df["days_since_last_holiday"] = df["date"].map(date_map).fillna(999).astype(int)
    return df


def engineer_features(
    input_parquet: Path = PROCESSED_DATA_DIR / "daily_demand.parquet",
    output_parquet: Path = PROCESSED_DATA_DIR / "features.parquet",
    group_col: str = "department",
    drop_na_lags: bool = True,
) -> pd.DataFrame:
    """
    End-to-end feature engineering pipeline for daily demand forecasting.

    Args:
        input_parquet (Path): Path to processed daily demand parquet table.
        output_parquet (Path): Destination path for engineered features table.
        group_col (str): Categorical grouping column ('department' or 'aisle').
        drop_na_lags (bool): Whether to drop initial rows with NaNs resulting from 28-day lags.

    Returns:
        pd.DataFrame: Final feature-engineered DataFrame.
    """
    if not input_parquet.exists():
        logger.info(f"Input file missing at '{input_parquet}'. Running data loader first...")
        df = load_and_process_demand(data_dir=RAW_DATA_DIR, output_file=input_parquet)
    else:
        logger.info(f"Loading daily demand table from '{input_parquet}'...")
        df = pd.read_parquet(input_parquet)

    # sort deterministically by group and date
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=[group_col, "date"]).reset_index(drop=True)

    # Add Lag Features
    df = add_lag_features(df, lags=[1, 7, 14, 28], group_col=group_col)

    # Add Rolling Features
    df = add_rolling_features(df, windows=[7, 28], group_col=group_col)

    # Add Calendar Features
    df = add_calendar_features(df)

    # Add Holiday Proximity Features
    df = add_holiday_features(df, holidays=INDIAN_HOLIDAYS)

    # drop initial rows where 28-day lag is NaN
    if drop_na_lags:
        initial_len = len(df)
        df = df.dropna(subset=["demand_lag_28"]).reset_index(drop=True)
        logger.info(f"Dropped {initial_len - len(df):,} initial rows with missing lag values.")

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving engineered features ({len(df):,} rows, {df.shape[1]} columns) to '{output_parquet}'...")
    df.to_parquet(output_parquet, index=False)
    logger.info("Feature engineering pipeline completed successfully!")

    return df


if __name__ == "__main__":
    features_df = engineer_features()
    print("\nEngineered Feature Preview:")
    print(features_df.head(5))
    print(f"\nFeature Columns ({len(features_df.columns)}):", list(features_df.columns))
