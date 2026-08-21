import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Union


import pandas as pd
import numpy as np

# Ensure project root is in sys.path when executed directly
file_path = Path(__file__).resolve()
root_dir = file_path.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
except ModuleNotFoundError:
    from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Optimal data types to reduce RAM usage for Instacart dataset
RAW_DTYPES = {
    "orders": {
        "order_id": "int32",
        "user_id": "int32",
        "eval_set": "category",
        "order_number": "int16",
        "order_dow": "int8",
        "order_hour_of_day": "int8",
        "days_since_prior_order": "float32",
    },
    "order_products": {
        "order_id": "int32",
        "product_id": "int32",
        "add_to_cart_order": "int16",
        "reordered": "int8",
    },
    "products": {
        "product_id": "int32",
        "product_name": "category",
        "aisle_id": "int16",
        "department_id": "int16",
    },
    "aisles": {
        "aisle_id": "int16",
        "aisle": "category",
    },
    "departments": {
        "department_id": "int16",
        "department": "category",
    },
}

REQUIRED_FILES = [
    "orders.csv",
    "order_products__prior.csv",
    "order_products__train.csv",
    "products.csv",
    "aisles.csv",
    "departments.csv",
]


def load_raw_data(data_dir: Path = RAW_DATA_DIR) -> Dict[str, pd.DataFrame]:
    """
    Loads Instacart raw CSV files into pandas DataFrames using optimized data types.

    Args:
        data_dir (Path): Path to directory containing raw CSV files.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping table name to DataFrame.

    Raises:
        FileNotFoundError: If any required CSV file is missing from data_dir.
    """
    data_dir = Path(data_dir)
    missing_files = [f for f in REQUIRED_FILES if not (data_dir / f).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required CSV files in {data_dir}: {', '.join(missing_files)}"
        )

    logger.info(f"Loading raw CSV datasets from '{data_dir}'...")

    orders = pd.read_csv(data_dir / "orders.csv", dtype=RAW_DTYPES["orders"])
    order_prior = pd.read_csv(data_dir / "order_products__prior.csv", dtype=RAW_DTYPES["order_products"])
    order_train = pd.read_csv(data_dir / "order_products__train.csv", dtype=RAW_DTYPES["order_products"])
    products = pd.read_csv(data_dir / "products.csv", dtype=RAW_DTYPES["products"])
    aisles = pd.read_csv(data_dir / "aisles.csv", dtype=RAW_DTYPES["aisles"])
    departments = pd.read_csv(data_dir / "departments.csv", dtype=RAW_DTYPES["departments"])

    # Combine prior and train order_products
    order_products = pd.concat([order_prior, order_train], ignore_index=True)
    logger.info(f"Loaded total {len(order_products):,} order-product items and {len(orders):,} orders.")

    return {
        "orders": orders,
        "order_products": order_products,
        "products": products,
        "aisles": aisles,
        "departments": departments,
    }


def create_flat_dataframe(raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merges order, product, aisle, and department tables into a single flat DataFrame.

    Args:
        raw_data (Dict[str, pd.DataFrame]): Raw data dictionary from load_raw_data().

    Returns:
        pd.DataFrame: Merged flat DataFrame with key attributes.
    """
    logger.info("Merging tables into a flat DataFrame...")

    # Join products with aisles and departments
    prod_meta = (
        raw_data["products"]
        .merge(raw_data["aisles"], on="aisle_id", how="left")
        .merge(raw_data["departments"], on="department_id", how="left")
    )

    # Join order_products with product metadata
    merged = raw_data["order_products"].merge(prod_meta, on="product_id", how="left")

    # Join with orders metadata
    flat_df = merged.merge(raw_data["orders"], on="order_id", how="left")

    # Required output columns
    selected_cols = [
        "order_id",
        "user_id",
        "product_id",
        "product_name",
        "aisle",
        "department",
        "order_dow",
        "order_hour_of_day",
        "days_since_prior_order",
    ]

    flat_df = flat_df[selected_cols]
    logger.info(f"Created flat DataFrame with shape {flat_df.shape}.")
    return flat_df


def add_synthetic_date(orders_df: pd.DataFrame, start_date: str = "2024-01-01") -> pd.DataFrame:
    """
    Constructs a continuous synthetic date column for orders by accumulating
    days_since_prior_order per user.

    Args:
        orders_df (pd.DataFrame): Orders metadata DataFrame containing user_id and days_since_prior_order.
        start_date (str): Base calendar date for the start of history.

    Returns:
        pd.DataFrame: Copy of orders DataFrame with added 'date' column.
    """
    df = orders_df.copy()
    # Replace NaN for initial orders with 0 days
    days_filled = df["days_since_prior_order"].fillna(0)
    
    # Cumulative days per user
    df["cum_days"] = days_filled.groupby(df["user_id"]).cumsum().astype(int)
    
    # Map to datetime
    base = pd.Timestamp(start_date)
    df["date"] = base + pd.to_timedelta(df["cum_days"], unit="D")
    return df


def aggregate_daily_demand(
    flat_df: pd.DataFrame,
    orders_df: Optional[pd.DataFrame] = None,
    level: str = "department",
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """
    Aggregates order items into daily demand counts per category level (department or aisle).

    Args:
        flat_df (pd.DataFrame): Flat merged DataFrame.
        orders_df (Optional[pd.DataFrame]): Orders DataFrame for synthetic date generation.
            If None, synthetic dates are generated directly from flat_df.
        level (str): Aggregation category level ('department' or 'aisle').
        start_date (str): Base date for synthetic temporal sequence.

    Returns:
        pd.DataFrame: Daily aggregated demand time series.
    """
    if level not in ["department", "aisle"]:
        raise ValueError("Aggregation level must be 'department' or 'aisle'.")

    logger.info(f"Aggregating daily demand by '{level}'...")
    
    df = flat_df.copy()
    if "date" not in df.columns:
        if orders_df is not None:
            dated_orders = add_synthetic_date(orders_df, start_date=start_date)
            df = df.merge(dated_orders[["order_id", "date"]], on="order_id", how="left")
        else:
            # Fallback: cumulative sum per user within flat_df
            df["days_filled"] = df["days_since_prior_order"].fillna(0)
            df["cum_days"] = df.groupby("user_id")["days_filled"].transform("cumsum").astype(int)
            base = pd.Timestamp(start_date)
            df["date"] = base + pd.to_timedelta(df["cum_days"], unit="D")

    # Aggregate demand
    daily = (
        df.groupby(["date", level, "order_dow"], observed=False)
        .agg(
            demand=("product_id", "count"),
            unique_orders=("order_id", "nunique"),
            unique_users=("user_id", "nunique"),
        )

        .reset_index()
        .sort_values(by=["date", level])
    )

    logger.info(f"Aggregated into {len(daily):,} daily demand records.")
    return daily


def load_and_process_demand(
    data_dir: Path = RAW_DATA_DIR,
    output_file: Path = PROCESSED_DATA_DIR / "daily_demand.parquet",
    level: str = "department",
) -> pd.DataFrame:
    """
    Full pipeline: Loads raw CSVs, creates flat DataFrame, aggregates daily demand,
    and exports result to Parquet.

    Args:
        data_dir (Path): Source folder for raw CSV files.
        output_file (Path): Output filepath for parquet table.
        level (str): Aggregation granularity level ('department' or 'aisle').

    Returns:
        pd.DataFrame: Final aggregated daily demand DataFrame.
    """
    raw_data = load_raw_data(data_dir=data_dir)
    flat_df = create_flat_dataframe(raw_data)
    daily_demand = aggregate_daily_demand(flat_df, orders_df=raw_data["orders"], level=level)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving processed daily demand dataset to '{output_path}'...")
    daily_demand.to_parquet(output_path, index=False)
    logger.info("Pipeline execution complete!")

    return daily_demand


if __name__ == "__main__":
    import sys
    try:
        df_result = load_and_process_demand()
        print("\nProcessed Daily Demand Preview:")
        print(df_result.head(10))
        print(f"\nTotal rows: {len(df_result):,}")
    except FileNotFoundError as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)