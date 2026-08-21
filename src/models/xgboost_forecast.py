import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error

file_path = Path(__file__).resolve()
root_dir = file_path.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
    from src.features.build_features import engineer_features
    from src.models.baseline_forecast import run_baseline_pipeline
except ModuleNotFoundError:
    from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
    from build_features import engineer_features
    from baseline_forecast import run_baseline_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Model output path
MODEL_PATH = MODELS_DIR / "xgboost_demand.pkl"


FEATURE_COLS = [
    "order_dow",
    "demand_lag_1",
    "demand_lag_7",
    "demand_lag_14",
    "demand_lag_28",
    "demand_rolling_mean_7",
    "demand_rolling_std_7",
    "demand_rolling_mean_28",
    "demand_rolling_std_28",
    "day_of_week",
    "is_weekend",
    "month",
    "days_since_last_holiday",
]


def load_feature_data(
    features_path: Path = PROCESSED_DATA_DIR / "features.parquet",
    department: str = "produce",
) -> pd.DataFrame:
    """
    Loads engineered features and filters data for the specified department.

    Args:
        features_path (Path): Path to features.parquet file.
        department (str): Target department.

    Returns:
        pd.DataFrame: Feature DataFrame for department.
    """
    if not features_path.exists():
        logger.info(f"Features file missing at {features_path}. Running feature engineering...")
        df = engineer_features(output_parquet=features_path)
    else:
        df = pd.read_parquet(features_path)

    df["date"] = pd.to_datetime(df["date"])
    dept_df = df[df["department"] == department].sort_values("date").reset_index(drop=True)
    logger.info(f"Loaded {len(dept_df)} feature rows for department '{department}'.")
    return dept_df


def train_test_split_time_based(
    df: pd.DataFrame, test_days: int = 28
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs chronological time-based train/test split without shuffling.

    Args:
        df (pd.DataFrame): Sorted feature DataFrame.
        test_days (int): Holdout test duration (days).

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    cutoff_date = df["date"].max() - pd.Timedelta(days=test_days - 1)
    train_df = df[df["date"] < cutoff_date].copy()
    test_df = df[df["date"] >= cutoff_date].copy()

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["demand"]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["demand"]

    logger.info(
        f"Time-based split: Train [{train_df['date'].min().date()} to {train_df['date'].max().date()}] ({len(train_df)} rows), "
        f"Test [{test_df['date'].min().date()} to {test_df['date'].max().date()}] ({len(test_df)} rows)."
    )
    return X_train, X_test, y_train, y_test, train_df, test_df


def tune_xgboost_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """
    Performs hyperparameter search using TimeSeriesSplit cross-validation.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Target demand.

    Returns:
        XGBRegressor: Best fitted XGBoost model.
    """
    logger.info("Starting XGBoost hyperparameter tuning with TimeSeriesSplit CV...")

    param_grid = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.1, "subsample": 0.9},
        {"n_estimators": 100, "max_depth": 7, "learning_rate": 0.1, "subsample": 0.9},
    ]

    tscv = TimeSeriesSplit(n_splits=3)
    best_score = float("inf")
    best_params = param_grid[0]

    for params in param_grid:
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = XGBRegressor(**params, random_state=42, n_jobs=1)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            scores.append(rmse)

        mean_score = np.mean(scores)
        if mean_score < best_score:
            best_score = mean_score
            best_params = params

    logger.info(f"Best Hyperparameters: {best_params} (CV RMSE: {best_score:.2f})")
    best_model = XGBRegressor(**best_params, random_state=42, n_jobs=1)
    best_model.fit(X_train, y_train)
    return best_model




def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates RMSE and MAPE evaluation metrics.

    Args:
        y_true (np.ndarray): Actual values.
        y_pred (np.ndarray): Forecast values.

    Returns:
        Dict[str, float]: RMSE and MAPE metrics.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"RMSE": round(rmse, 2), "MAPE (%)": round(mape, 2)}


def plot_feature_importance(
    model: XGBRegressor,
    feature_names: list,
    output_plot_path: Path = Path("reports/xgboost_feature_importance.png"),
):
    """
    Plots gain feature importances for the trained XGBoost model.

    Args:
        model (XGBRegressor): Fitted XGBoost model.
        feature_names (list): List of feature names.
        output_plot_path (Path): Filepath to save generated plot.
    """
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"], importance_df["importance"], color="#2b5c8f")
    plt.title("XGBoost Model Feature Importance (Gain)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Relative Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()

    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=300)
    logger.info(f"Saved feature importance chart to '{output_plot_path}'.")
    plt.close()


def plot_actual_vs_xgb(
    test_df: pd.DataFrame,
    xgb_preds: np.ndarray,
    department: str,
    output_plot_path: Path = Path("reports/xgboost_vs_actual.png"),
):
    """
    Plots Actual vs XGBoost predicted demand over the holdout test period.

    Args:
        test_df (pd.DataFrame): Test DataFrame.
        xgb_preds (np.ndarray): XGBoost predictions.
        department (str): Department name.
        output_plot_path (Path): Chart export destination.
    """
    plt.figure(figsize=(13, 6))
    dates = test_df["date"]
    actuals = test_df["demand"].values

    plt.plot(dates, actuals, label="Actual Demand", color="black", linewidth=2.5, marker="o")
    plt.plot(dates, xgb_preds, label="XGBoost Forecast", color="#2ca02c", linestyle="--", linewidth=2.0, marker="s")

    plt.title(f"XGBoost Demand Forecast ({department.title()} Department - 28-Day Test Set)", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Daily Item Demand", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.xticks(rotation=30)
    plt.tight_layout()

    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=300)
    logger.info(f"Saved XGBoost forecast plot to '{output_plot_path}'.")
    plt.close()



def run_xgboost_pipeline(
    department: str = "produce",
    test_days: int = 28,
    model_output_path: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Runs full XGBoost pipeline: loading features, hyperparameter search, holdout testing,
    evaluation against baselines, plotting feature importances, and saving model artifact.

    Args:
        department (str): Target department.
        test_days (int): Holdout test horizon in days.
        model_output_path (Optional[Path]): Target model export file path.

    Returns:
        Dict[str, float]: XGBoost test evaluation metrics.
    """
    if model_output_path is None:
        model_output_path = MODELS_DIR / "xgboost_demand.pkl"

    df = load_feature_data(department=department)
    X_train, X_test, y_train, y_test, train_df, test_df = train_test_split_time_based(df, test_days=test_days)


    # Hyperparameter Tuning
    best_xgb = tune_xgboost_hyperparameters(X_train, y_train)

    # Evaluate Forecast
    xgb_preds = best_xgb.predict(X_test)
    xgb_metrics = calculate_metrics(y_test.values, xgb_preds)

    logger.info(f"=== XGBoost Evaluation Results ({department.title()}) ===")
    logger.info(f"XGBoost -> RMSE: {xgb_metrics['RMSE']}, MAPE: {xgb_metrics['MAPE (%)']}%")

    # Plot Feature Importance & Predictions
    plot_feature_importance(best_xgb, FEATURE_COLS)
    plot_actual_vs_xgb(test_df, xgb_preds, department=department)

    # Export Model Artifact
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_xgb, model_output_path)
    logger.info(f"Saved trained XGBoost model artifact to '{model_output_path}'.")

    return xgb_metrics


if __name__ == "__main__":
    run_xgboost_pipeline(department="produce", test_days=28)
