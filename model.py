"""
model.py - Demand Forecasting Engine using Scikit-Learn

This module trains a machine learning demand forecasting model
(RandomForestRegressor) on retail sales history.

Features:
- day_of_week, month, price, promotion
- lag_7, lag_14 (historical demand at 7 and 14 days prior)
- rolling_mean_7, rolling_mean_30 (recent rolling consumption)
- product_code, category_code

Evaluation:
- Time-based train/test split (unseen evaluation window)
- Compares MAE and RMSE against a Moving Average baseline
- Serializes trained model and metadata using joblib
- Generates rolling autoregressive multi-step future forecasts (next 30 days)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL_FILE = "demand_model.joblib"
DATA_FILE = "retail_inventory_data.csv"
RANDOM_SEED = 42


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs temporal, promotional, lag, and rolling statistical features.
    Carefully shifts by 1 day before rolling calculations to prevent data leakage.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

    # 1. Calendar / temporal features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # 2. Categorical encodings
    if "product_code" not in df.columns:
        df["product_code"] = df["product_id"].astype("category").cat.codes
    if "category_code" not in df.columns:
        df["category_code"] = df["category"].astype("category").cat.codes

    # 3. Lag features (per product)
    df["lag_7"] = df.groupby("product_id")["sales"].shift(7)
    df["lag_14"] = df.groupby("product_id")["sales"].shift(14)

    # 4. Rolling statistical features (lagged by 1 day to ensure zero leakage)
    df["rolling_mean_7"] = df.groupby("product_id")["sales"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )
    df["rolling_mean_30"] = df.groupby("product_id")["sales"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=1).mean()
    )

    return df


def train_demand_model(data_source=DATA_FILE,
                       test_days: int = 60,
                       save_path: str = MODEL_FILE):
    """
    Trains a RandomForest demand forecasting model using a time-based train/test split.
    Accepts either a filepath or a pandas DataFrame.
    Prints MAE and RMSE metrics compared to a Moving Average baseline.
    Saves model and metadata using joblib.
    """
    if isinstance(data_source, pd.DataFrame):
        df = data_source.copy()
    elif isinstance(data_source, str):
        if not os.path.exists(data_source):
            from data_generator import generate_sales_data
            print(f"Data file '{data_source}' not found. Generating now...")
            generate_sales_data(output_file=data_source)
        df = pd.read_csv(data_source)
    else:
        df = pd.read_csv(data_source)

    df["date"] = pd.to_datetime(df["date"])

    # Create features
    df_feat = create_features(df)

    # Drop rows where lag features are NaN (first 14 days)
    df_clean = df_feat.dropna(subset=["lag_7", "lag_14", "rolling_mean_7", "rolling_mean_30"]).copy()

    # Adaptive time-based split for datasets of varying sizes
    max_date = df_clean["date"].max()
    min_date = df_clean["date"].min()
    total_days = max(1, (max_date - min_date).days)
    effective_test_days = min(test_days, max(7, int(total_days * 0.2)))
    split_date = max_date - pd.Timedelta(days=effective_test_days)

    train_df = df_clean[df_clean["date"] <= split_date].copy()
    test_df = df_clean[df_clean["date"] > split_date].copy()

    if train_df.empty or test_df.empty:
        # Fallback if dataset is very small: use 80/20 random or chronological row split
        split_idx = int(len(df_clean) * 0.8)
        train_df = df_clean.iloc[:split_idx].copy()
        test_df = df_clean.iloc[split_idx:].copy()

    feature_cols = [
        "product_code",
        "category_code",
        "day_of_week",
        "month",
        "price",
        "promotion",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_mean_30"
    ]
    target_col = "sales"

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Train Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=4,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred = np.maximum(0, np.round(y_pred))  # Sales cannot be negative
    test_df["predicted_sales"] = y_pred

    # ML Model Metrics
    ml_mae = mean_absolute_error(y_test, y_pred)
    ml_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Simple Baseline Model: 7-day Moving Average (already present in rolling_mean_7)
    baseline_pred = test_df["rolling_mean_7"].fillna(test_df["sales"].mean()).values
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

    # Improvement calculation
    mae_improvement = ((baseline_mae - ml_mae) / baseline_mae) * 100.0 if baseline_mae > 0 else 0.0
    rmse_improvement = ((baseline_rmse - ml_rmse) / baseline_rmse) * 100.0 if baseline_rmse > 0 else 0.0

    print("=" * 60)
    print("        OPTIMISED RETAIL DEMAND FORECASTING BENCHMARK       ")
    print("=" * 60)
    print(f"Training Range: {train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')} ({len(train_df)} rows)")
    print(f"Testing Range:  {test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')} ({len(test_df)} rows)")
    print("-" * 60)
    print(f"Moving Average Baseline: MAE = {baseline_mae:.2f} | RMSE = {baseline_rmse:.2f}")
    print(f"Random Forest ML Model:  MAE = {ml_mae:.2f} | RMSE = {ml_rmse:.2f}")
    print(f"Performance Gain:        MAE {mae_improvement:+.1f}% | RMSE {rmse_improvement:+.1f}%")
    print("=" * 60)

    # Save model, encoders, and evaluation metadata
    product_mapping = dict(zip(df_clean["product_id"], df_clean["product_code"]))
    category_mapping = dict(zip(df_clean["category"], df_clean["category_code"]))

    payload = {
        "model": model,
        "feature_cols": feature_cols,
        "product_mapping": product_mapping,
        "category_mapping": category_mapping,
        "metrics": {
            "ml_mae": round(ml_mae, 3),
            "ml_rmse": round(ml_rmse, 3),
            "baseline_mae": round(baseline_mae, 3),
            "baseline_rmse": round(baseline_rmse, 3),
            "mae_improvement_pct": round(mae_improvement, 1),
            "rmse_improvement_pct": round(rmse_improvement, 1),
            "split_date": split_date.strftime("%Y-%m-%d") if hasattr(split_date, "strftime") else str(split_date),
            "max_date": max_date.strftime("%Y-%m-%d")
        },
        "test_df": test_df[["date", "product_id", "product_name", "category", "sales", "predicted_sales", "rolling_mean_7"]]
    }

    if save_path:
        joblib.dump(payload, save_path)
        print(f"Saved trained model package to: {save_path}")
    return payload


def load_demand_model(model_path: str = MODEL_FILE):
    """Loads saved model payload or trains it if not present."""
    if not os.path.exists(model_path):
        return train_demand_model(save_path=model_path)
    return joblib.load(model_path)


def forecast_next_days(df_historical: pd.DataFrame,
                       model_payload: dict,
                       forecast_horizon_days: int = 30) -> pd.DataFrame:
    """
    Generates dynamic multi-step rolling forecast for each product for the next N days.
    Dynamically rolls forward lags and moving averages as future days are predicted.
    Robust against unseen products and categories.
    """
    model = model_payload["model"]
    feature_cols = model_payload["feature_cols"]
    prod_map = model_payload.get("product_mapping", {})
    cat_map = model_payload.get("category_mapping", {})

    df = df_historical.copy()
    df["date"] = pd.to_datetime(df["date"])
    max_date = df["date"].max()

    # Get latest product metadata (price, category, etc.)
    latest_meta = df.sort_values("date").groupby("product_id").last().reset_index()

    future_records = []

    # Maintain recent sales history per product for rolling calculations
    history_dict = {}
    for pid in latest_meta["product_id"].unique():
        prod_data = df[df["product_id"] == pid].sort_values("date")
        sales_vals = list(prod_data["sales"].values)
        history_dict[pid] = sales_vals if len(sales_vals) > 0 else [20.0]

    future_dates = [max_date + pd.Timedelta(days=i) for i in range(1, forecast_horizon_days + 1)]

    for dt in future_dates:
        dow = dt.dayofweek
        month = dt.month
        # Weekend slight promotion probability
        promo = 1 if (dow in [5, 6] and np.random.rand() < 0.20) else 0

        for _, meta in latest_meta.iterrows():
            pid = meta["product_id"]
            pname = meta.get("product_name", pid)
            cat = meta.get("category", "General")
            p_code = prod_map.get(pid, 0)
            c_code = cat_map.get(cat, 0)
            base_price = float(meta.get("price", 25.0))
            price = round(base_price * 0.85, 2) if promo == 1 else base_price

            past_sales = history_dict[pid]
            # lag 7 is sales 7 days ago, lag 14 is sales 14 days ago
            lag_7 = past_sales[-7] if len(past_sales) >= 7 else np.mean(past_sales)
            lag_14 = past_sales[-14] if len(past_sales) >= 14 else np.mean(past_sales)
            roll_7 = np.mean(past_sales[-7:]) if len(past_sales) >= 7 else np.mean(past_sales)
            roll_30 = np.mean(past_sales[-30:]) if len(past_sales) >= 30 else np.mean(past_sales)

            row_features = pd.DataFrame([{
                "product_code": p_code,
                "category_code": c_code,
                "day_of_week": dow,
                "month": month,
                "price": price,
                "promotion": promo,
                "lag_7": lag_7,
                "lag_14": lag_14,
                "rolling_mean_7": roll_7,
                "rolling_mean_30": roll_30
            }])[feature_cols]

            pred_val = float(model.predict(row_features)[0])
            pred_val = max(0.0, round(pred_val, 1))

            # Append to rolling history
            past_sales.append(pred_val)

            future_records.append({
                "date": dt.strftime("%Y-%m-%d"),
                "product_id": pid,
                "product_name": pname,
                "category": cat,
                "predicted_sales": pred_val,
                "price": price,
                "promotion": promo
            })

    return pd.DataFrame(future_records)


if __name__ == "__main__":
    train_demand_model()
