import sys
import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    Prophet = None


file_path = Path(__file__).resolve()
root_dir = file_path.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
    from src.data_loader import load_and_process_demand
except ModuleNotFoundError:
    from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
    from data_loader import load_and_process_demand


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_department_series(
    parquet_path: Path = PROCESSED_DATA_DIR / "daily_demand.parquet",
    department: str = "produce",
) -> pd.DataFrame:
    """
    Loads daily demand dataset and extracts time series for a single department.

    Args:
        parquet_path (Path): Path to daily demand parquet table.
        department (str): Target department to forecast.

    Returns:
        pd.DataFrame: Sorted daily demand dataframe for the selected department.
    """
    if not parquet_path.exists():
        logger.info(f"Parquet dataset missing at {parquet_path}. Running data loader pipeline...")
        df = load_and_process_demand(data_dir=RAW_DATA_DIR, output_file=parquet_path)
    else:
        df = pd.read_parquet(parquet_path)

    # Aggregate department daily total demand 
    dept_df = (
        df[df["department"] == department]
        .groupby("date")["demand"]
        .sum()
        .reset_index()
        .sort_values("date")
    )

    if len(dept_df) == 0:
        available_depts = df["department"].unique()
        raise ValueError(f"Department '{department}' not found. Available: {list(available_depts)}")

    logger.info(f"Loaded {len(dept_df)} days of demand data for department '{department}'.")
    return dept_df


def train_test_split_by_date(
    df: pd.DataFrame, test_days: int = 28
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits time series data into train and test sets based on chronological cutoff.

    Args:
        df (pd.DataFrame): Time series DataFrame with 'date' and 'demand'.
        test_days (int): Number of holdout days for testing (default: 28 days / 4 weeks).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    cutoff_date = df["date"].max() - pd.Timedelta(days=test_days - 1)
    train_df = df[df["date"] < cutoff_date].copy()
    test_df = df[df["date"] >= cutoff_date].copy()

    logger.info(
        f"Train/Test split: Train range [{train_df['date'].min().date()} to {train_df['date'].max().date()}] ({len(train_df)} days), "
        f"Test range [{test_df['date'].min().date()} to {test_df['date'].max().date()}] ({len(test_df)} days)."
    )
    return train_df, test_df


import joblib
import json

def fit_sarima_forecast(
    train_df: pd.DataFrame, test_days: int, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), save_model: bool = True
) -> np.ndarray:
    """
    Fits a SARIMA model using statsmodels, forecasts test_days ahead, and exports fitted model artifact.

    Args:
        train_df (pd.DataFrame): Training data.
        test_days (int): Forecast horizon.
        order (tuple): Non-seasonal ARIMA order (p, d, q).
        seasonal_order (tuple): Seasonal ARIMA order (P, D, Q, s).
        save_model (bool): Whether to save binary model to models/sarima_model.pkl.

    Returns:
        np.ndarray: Predicted demand array.
    """
    logger.info(f"Fitting SARIMA{order}x{seasonal_order} model...")
    ts_train = train_df.set_index("date")["demand"].asfreq("D")
    
    # Fill any missing dates 
    ts_train = ts_train.ffill().bfill()

    model = SARIMAX(
        ts_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted_model = model.fit(disp=False)
    
    if save_model:
        sarima_pkl_path = MODELS_DIR / "sarima_model.pkl"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted_model, sarima_pkl_path)
        logger.info(f"Saved SARIMA baseline model artifact to '{sarima_pkl_path}'.")

    forecast_vals = fitted_model.forecast(steps=test_days)
    return forecast_vals.values



def fit_prophet_forecast(train_df: pd.DataFrame, test_days: int) -> np.ndarray:
    """
    Fits a Prophet model and forecasts test_days ahead.
    Falls back to Seasonal Naive (7-day lag) if Prophet is not installed.

    Args:
        train_df (pd.DataFrame): Training DataFrame with 'date' and 'demand'.
        test_days (int): Forecast horizon.

    Returns:
        np.ndarray: Predicted demand array for test period.
    """
    if not HAS_PROPHET:
        logger.warning(
            "Prophet library is not installed. Using 7-day Seasonal Naive baseline as fallback."
        )
        # repeating the last 7 days of training set
        last_7_days = train_df["demand"].tail(7).values
        repeated = np.tile(last_7_days, int(np.ceil(test_days / 7)))[:test_days]
        return repeated

    logger.info("Fitting Prophet baseline model...")
    prophet_train = train_df[["date", "demand"]].rename(columns={"date": "ds", "demand": "y"})

    m = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95,
    )
    m.fit(prophet_train)

    future = m.make_future_dataframe(periods=test_days, freq="D")
    forecast = m.predict(future)

    test_preds = forecast.tail(test_days)["yhat"].values
    return test_preds



def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates RMSE and MAPE evaluation metrics.

    Args:
        y_true (np.ndarray): Actual ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        Dict[str, float]: Dictionary with RMSE and MAPE.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"RMSE": round(rmse, 2), "MAPE (%)": round(mape, 2)}


def plot_baseline_comparison(
    test_df: pd.DataFrame,
    sarima_preds: np.ndarray,
    prophet_preds: np.ndarray,
    department: str,
    output_plot_path: Path = Path("reports/baseline_forecast_comparison.png"),
):
    """
    Plots Actual vs SARIMA vs Prophet forecasts over the test holdout set.

    Args:
        test_df (pd.DataFrame): Test DataFrame containing dates and actual demand.
        sarima_preds (np.ndarray): Predictions from SARIMA.
        prophet_preds (np.ndarray): Predictions from Prophet.
        department (str): Name of department.
        output_plot_path (Path): Path to save comparison plot image.
    """
    plt.figure(figsize=(13, 6))
    dates = test_df["date"]
    actuals = test_df["demand"].values

    plt.plot(dates, actuals, label="Actual Demand", color="black", linewidth=2.5, marker="o")
    plt.plot(dates, sarima_preds, label="SARIMA Forecast", color="#1f77b4", linestyle="--", linewidth=2.0)
    plt.plot(dates, prophet_preds, label="Prophet Forecast", color="#d95f02", linestyle="-.", linewidth=2.0)

    plt.title(f"Baseline Models Forecast Comparison ({department.title()} Department - 28-Day Test Set)", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Daily Item Demand", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=11)
    plt.xticks(rotation=30)
    plt.tight_layout()

    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=300)
    logger.info(f"Saved forecast comparison plot to '{output_plot_path}'.")
    plt.close()



def run_baseline_pipeline(
    department: str = "produce",
    test_days: int = 28,
    output_plot_path: Path = Path("reports/baseline_forecast_comparison.png"),
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Runs end-to-end baseline comparison pipeline for selected department.

    Args:
        department (str): Target department.
        test_days (int): Holdout test horizon (days).
        output_plot_path (Path): Filepath to save generated chart.

    Returns:
        Tuple[Dict[str, float], Dict[str, float]]: (sarima_metrics, prophet_metrics)
    """
    df = load_department_series(department=department)
    train_df, test_df = train_test_split_by_date(df, test_days=test_days)

    actuals = test_df["demand"].values

    # Fit SARIMA
    sarima_preds = fit_sarima_forecast(train_df, test_days=test_days)
    sarima_metrics = calculate_metrics(actuals, sarima_preds)

    # Fit Prophet
    prophet_preds = fit_prophet_forecast(train_df, test_days=test_days)
    prophet_metrics = calculate_metrics(actuals, prophet_preds)

    logger.info(f"--- Evaluation Results ({department.title()} Department) ---")
    logger.info(f"SARIMA  -> RMSE: {sarima_metrics['RMSE']}, MAPE: {sarima_metrics['MAPE (%)']}%")
    logger.info(f"Prophet -> RMSE: {prophet_metrics['RMSE']}, MAPE: {prophet_metrics['MAPE (%)']}%")

    # Export baseline evaluation metrics artifact
    metrics_json_path = MODELS_DIR / "baseline_metrics.json"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(metrics_json_path, "w") as f:
        json.dump({
            "department": department,
            "sarima": sarima_metrics,
            "prophet": prophet_metrics,
        }, f, indent=2)
    logger.info(f"Saved baseline evaluation metrics to '{metrics_json_path}'.")


    # comparison visualization
    plot_baseline_comparison(
        test_df,
        sarima_preds,
        prophet_preds,
        department=department,
        output_plot_path=output_plot_path,
    )

    return sarima_metrics, prophet_metrics


if __name__ == "__main__":
    run_baseline_pipeline(department="produce", test_days=28)
