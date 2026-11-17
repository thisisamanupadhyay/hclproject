"""
data_generator.py - Realistic Synthetic Retail Sales & Inventory Generator

Reads product metadata from a CSV catalog (e.g. products_catalog.csv) and generates
daily sales transaction records with realistic seasonality, pricing, and supply-chain parameters.

Features modelled:
- Reads product catalog dynamically from CSV (no need to write manually in code)
- Weekly seasonality (higher sales on weekends / Fridays)
- Monthly & holiday spikes (Q4 holiday surge, summer discount surge)
- Dynamic pricing and promotional discounts
- Operational supply chain parameters (lead time, unit costs, holding and ordering costs)
- Realistic current stock levels for inventory simulation
"""

import os
import argparse
import random
import numpy as np
import pandas as pd

# Set fixed random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEFAULT_CATALOG_FILE = "products_catalog.csv"

# Fallback default catalog definitions in case CSV file is missing
DEFAULT_PRODUCTS_CATALOG = [
    # Electronics
    {"product_id": "P01", "product_name": "Wireless Optical Mouse", "category": "Electronics",
     "base_price": 25.0, "unit_cost": 14.0, "lead_time_days": 5, "holding_cost": 2.50, "ordering_cost": 45.0,
     "base_demand": 38, "current_stock": 140},
    {"product_id": "P02", "product_name": "Mechanical Gaming Keyboard", "category": "Electronics",
     "base_price": 85.0, "unit_cost": 50.0, "lead_time_days": 8, "holding_cost": 8.00, "ordering_cost": 60.0,
     "base_demand": 24, "current_stock": 290},
    {"product_id": "P03", "product_name": "Ultra HD 4K Webcam", "category": "Electronics",
     "base_price": 65.0, "unit_cost": 38.0, "lead_time_days": 7, "holding_cost": 6.00, "ordering_cost": 55.0,
     "base_demand": 20, "current_stock": 120},
    {"product_id": "P04", "product_name": "Noise-Cancelling Headphones", "category": "Electronics",
     "base_price": 120.0, "unit_cost": 70.0, "lead_time_days": 10, "holding_cost": 12.00, "ordering_cost": 75.0,
     "base_demand": 18, "current_stock": 620},
    {"product_id": "P05", "product_name": "Multi-Port USB-C Hub", "category": "Electronics",
     "base_price": 35.0, "unit_cost": 20.0, "lead_time_days": 6, "holding_cost": 3.50, "ordering_cost": 50.0,
     "base_demand": 35, "current_stock": 310},
    {"product_id": "P06", "product_name": "Portable Bluetooth Speaker", "category": "Electronics",
     "base_price": 45.0, "unit_cost": 26.0, "lead_time_days": 7, "holding_cost": 4.50, "ordering_cost": 50.0,
     "base_demand": 28, "current_stock": 280},
    {"product_id": "P07", "product_name": "Smart Fitness Watch", "category": "Electronics",
     "base_price": 95.0, "unit_cost": 55.0, "lead_time_days": 9, "holding_cost": 9.50, "ordering_cost": 65.0,
     "base_demand": 22, "current_stock": 160},

    # Groceries
    {"product_id": "P08", "product_name": "Organic Arabica Coffee Beans 1kg", "category": "Groceries",
     "base_price": 22.0, "unit_cost": 12.0, "lead_time_days": 4, "holding_cost": 2.20, "ordering_cost": 40.0,
     "base_demand": 55, "current_stock": 450},
    {"product_id": "P09", "product_name": "Japanese Matcha Green Tea 250g", "category": "Groceries",
     "base_price": 18.0, "unit_cost": 10.0, "lead_time_days": 5, "holding_cost": 1.80, "ordering_cost": 40.0,
     "base_demand": 32, "current_stock": 130},
    {"product_id": "P10", "product_name": "Almond Milk Unsweetened 1L", "category": "Groceries",
     "base_price": 4.5, "unit_cost": 2.4, "lead_time_days": 3, "holding_cost": 0.50, "ordering_cost": 35.0,
     "base_demand": 65, "current_stock": 1300},
    {"product_id": "P11", "product_name": "Extra Virgin Olive Oil 1L", "category": "Groceries",
     "base_price": 16.0, "unit_cost": 9.5, "lead_time_days": 6, "holding_cost": 1.60, "ordering_cost": 45.0,
     "base_demand": 40, "current_stock": 380},
    {"product_id": "P12", "product_name": "Organic Rolled Oats 1.5kg", "category": "Groceries",
     "base_price": 8.5, "unit_cost": 4.8, "lead_time_days": 4, "holding_cost": 0.90, "ordering_cost": 35.0,
     "base_demand": 48, "current_stock": 390},
    {"product_id": "P13", "product_name": "Creamy Peanut Butter 1kg", "category": "Groceries",
     "base_price": 9.0, "unit_cost": 5.0, "lead_time_days": 4, "holding_cost": 0.90, "ordering_cost": 35.0,
     "base_demand": 42, "current_stock": 360},
    {"product_id": "P14", "product_name": "85% Dark Artisan Chocolate 100g", "category": "Groceries",
     "base_price": 5.0, "unit_cost": 2.8, "lead_time_days": 4, "holding_cost": 0.50, "ordering_cost": 35.0,
     "base_demand": 50, "current_stock": 180},

    # Apparel
    {"product_id": "P15", "product_name": "Classic Cotton Crewneck T-Shirt", "category": "Apparel",
     "base_price": 24.0, "unit_cost": 11.0, "lead_time_days": 7, "holding_cost": 2.40, "ordering_cost": 45.0,
     "base_demand": 48, "current_stock": 480},
    {"product_id": "P16", "product_name": "Slim-Fit Stretch Denim Jeans", "category": "Apparel",
     "base_price": 55.0, "unit_cost": 26.0, "lead_time_days": 10, "holding_cost": 5.50, "ordering_cost": 60.0,
     "base_demand": 26, "current_stock": 250},
    {"product_id": "P17", "product_name": "Lightweight Running Shoes", "category": "Apparel",
     "base_price": 75.0, "unit_cost": 38.0, "lead_time_days": 12, "holding_cost": 7.50, "ordering_cost": 70.0,
     "base_demand": 22, "current_stock": 230},
    {"product_id": "P18", "product_name": "Merino Wool Winter Sweater", "category": "Apparel",
     "base_price": 68.0, "unit_cost": 34.0, "lead_time_days": 11, "holding_cost": 6.80, "ordering_cost": 65.0,
     "base_demand": 18, "current_stock": 580},
    {"product_id": "P19", "product_name": "Breathable Athletic Gym Hoodie", "category": "Apparel",
     "base_price": 48.0, "unit_cost": 24.0, "lead_time_days": 8, "holding_cost": 4.80, "ordering_cost": 50.0,
     "base_demand": 30, "current_stock": 320},
    {"product_id": "P20", "product_name": "All-Weather Windbreaker Jacket", "category": "Apparel",
     "base_price": 80.0, "unit_cost": 42.0, "lead_time_days": 12, "holding_cost": 8.00, "ordering_cost": 70.0,
     "base_demand": 16, "current_stock": 210},
]


def create_default_catalog_file(catalog_path: str = DEFAULT_CATALOG_FILE):
    """Creates the default products_catalog.csv file if it does not already exist."""
    df = pd.DataFrame(DEFAULT_PRODUCTS_CATALOG)
    df.to_csv(catalog_path, index=False)
    print(f"Created default product catalog CSV at: {catalog_path}")
    return df


def load_products_catalog(catalog_source=DEFAULT_CATALOG_FILE) -> pd.DataFrame:
    """
    Loads product catalog from a CSV filepath, file buffer, or pandas DataFrame.
    Gracefully cleans and validates columns with sensible fallbacks for any missing attributes.

    Parameters:
        catalog_source: str (filepath), file-like object, or pd.DataFrame

    Returns:
        pd.DataFrame: Cleaned catalog dataframe
    """
    if isinstance(catalog_source, pd.DataFrame):
        df = catalog_source.copy()
    elif hasattr(catalog_source, "read"):
        # File-like object (e.g., Streamlit UploadedFile)
        df = pd.read_csv(catalog_source)
    elif isinstance(catalog_source, str):
        if not os.path.exists(catalog_source):
            if catalog_source == DEFAULT_CATALOG_FILE:
                df = create_default_catalog_file(catalog_source)
            else:
                raise FileNotFoundError(f"Catalog file not found: {catalog_source}")
        else:
            df = pd.read_csv(catalog_source)
    else:
        # Fallback to default catalog
        df = pd.DataFrame(DEFAULT_PRODUCTS_CATALOG)

    # Standardize column names (lowercase & stripped)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Map alternative column names if present
    column_aliases = {
        "sku": "product_id",
        "id": "product_id",
        "item_id": "product_id",
        "name": "product_name",
        "item_name": "product_name",
        "title": "product_name",
        "dept": "category",
        "department": "category",
        "price": "base_price",
        "selling_price": "base_price",
        "cost": "unit_cost",
        "lead_time": "lead_time_days",
        "stock": "current_stock",
        "inventory": "current_stock",
        "demand": "base_demand",
        "avg_demand": "base_demand",
    }
    for alias, target in column_aliases.items():
        if alias in df.columns and target not in df.columns:
            df[target] = df[alias]

    # Ensure required columns with intelligent defaults
    if "product_id" not in df.columns:
        df["product_id"] = [f"SKU{i+1:02d}" for i in range(len(df))]
    if "product_name" not in df.columns:
        df["product_name"] = df["product_id"].astype(str)
    if "category" not in df.columns:
        df["category"] = "General"
    if "base_price" not in df.columns:
        df["base_price"] = 25.0
    if "unit_cost" not in df.columns:
        df["unit_cost"] = df["base_price"] * 0.55
    if "lead_time_days" not in df.columns:
        df["lead_time_days"] = 7
    if "holding_cost" not in df.columns:
        df["holding_cost"] = (df["base_price"] * 0.10).round(2).clip(lower=0.5)
    if "ordering_cost" not in df.columns:
        df["ordering_cost"] = 45.0
    if "base_demand" not in df.columns:
        df["base_demand"] = 30
    if "current_stock" not in df.columns:
        df["current_stock"] = (df["base_demand"] * 7).astype(int)

    # Cast to appropriate types
    df["product_id"] = df["product_id"].astype(str)
    df["product_name"] = df["product_name"].astype(str)
    df["category"] = df["category"].astype(str)
    df["base_price"] = pd.to_numeric(df["base_price"], errors="coerce").fillna(25.0)
    df["unit_cost"] = pd.to_numeric(df["unit_cost"], errors="coerce").fillna(df["base_price"] * 0.55)
    df["lead_time_days"] = pd.to_numeric(df["lead_time_days"], errors="coerce").fillna(7).astype(int)
    df["holding_cost"] = pd.to_numeric(df["holding_cost"], errors="coerce").fillna(df["base_price"] * 0.10).round(2)
    df["ordering_cost"] = pd.to_numeric(df["ordering_cost"], errors="coerce").fillna(45.0)
    df["base_demand"] = pd.to_numeric(df["base_demand"], errors="coerce").fillna(30).astype(int)
    df["current_stock"] = pd.to_numeric(df["current_stock"], errors="coerce").fillna(100).astype(int)

    return df


def get_products_catalog(catalog_source=DEFAULT_CATALOG_FILE) -> pd.DataFrame:
    """Returns product catalog as a pandas DataFrame from CSV or memory."""
    return load_products_catalog(catalog_source)


def generate_sales_data(catalog_source=DEFAULT_CATALOG_FILE,
                        output_file="retail_inventory_data.csv",
                        start_date="2024-01-01",
                        days=730):
    """
    Generates synthetic daily sales transaction records over a specified duration
    using product definitions loaded directly from a catalog CSV.

    Parameters:
        catalog_source: Filepath, buffer, or DataFrame of product catalog.
        output_file (str or None): Destination CSV path. If None, does not write to disk.
        start_date (str): Beginning date in YYYY-MM-DD.
        days (int): Number of days of historical data (default: 730 = 2 years).

    Returns:
        pd.DataFrame: Completed synthetic sales dataframe.
    """
    catalog_df = load_products_catalog(catalog_source)
    products_catalog = catalog_df.to_dict(orient="records")

    date_range = pd.date_range(start=start_date, periods=days, freq="D")
    records = []

    # Weekly seasonality factors (0 = Monday, ..., 6 = Sunday)
    day_multipliers = {
        0: 0.88,  # Monday
        1: 0.90,  # Tuesday
        2: 0.94,  # Wednesday
        3: 0.98,  # Thursday
        4: 1.22,  # Friday shopping boost
        5: 1.38,  # Saturday peak
        6: 1.25,  # Sunday family shopping
    }

    # Category monthly factors (1 = Jan, ..., 12 = Dec)
    # Generic fallback handles any arbitrary categories added by users
    default_monthly = {
        1: 1.00, 2: 0.95, 3: 0.98, 4: 1.00, 5: 1.02, 6: 1.05,
        7: 1.10, 8: 1.02, 9: 1.02, 10: 1.10, 11: 1.30, 12: 1.45
    }

    monthly_multipliers = {
        "Electronics": {
            1: 0.92, 2: 0.90, 3: 0.95, 4: 0.96, 5: 1.00, 6: 1.05,
            7: 1.15, 8: 1.02, 9: 1.08, 10: 1.15, 11: 1.45, 12: 1.60
        },
        "Groceries": {
            1: 1.02, 2: 0.96, 3: 0.98, 4: 1.00, 5: 1.01, 6: 1.04,
            7: 1.06, 8: 1.02, 9: 1.03, 10: 1.08, 11: 1.20, 12: 1.35
        },
        "Apparel": {
            1: 0.95, 2: 0.90, 3: 0.96, 4: 1.02, 5: 1.08, 6: 1.12,
            7: 1.20, 8: 1.05, 9: 1.00, 10: 1.12, 11: 1.40, 12: 1.55
        }
    }

    for current_date in date_range:
        weekday = current_date.weekday()
        month = current_date.month

        for p in products_catalog:
            cat = p["category"]
            base_demand = p["base_demand"]
            base_price = p["base_price"]

            # Seasonality multipliers with safe fallback for any custom category
            dow_factor = day_multipliers[weekday]
            cat_multipliers = monthly_multipliers.get(cat, default_monthly)
            month_factor = cat_multipliers.get(month, 1.0)

            # Product promotion: ~13% chance of a discount event
            promo = 1 if np.random.rand() < 0.13 else 0

            if promo == 1:
                promo_factor = np.random.uniform(1.35, 1.65)
                # Price discount between 10% to 20%
                discount = np.random.uniform(0.10, 0.20)
                actual_price = round(base_price * (1.0 - discount), 2)
            else:
                promo_factor = 1.00
                # Normal minor pricing fluctuation (+/- 2%)
                actual_price = round(base_price * np.random.uniform(0.98, 1.02), 2)

            # Expected demand
            expected_demand = base_demand * dow_factor * month_factor * promo_factor

            # Stochastic variance around the expectation
            noise = np.random.normal(0, max(2.0, expected_demand * 0.08))
            daily_sales = int(max(0, np.round(expected_demand + noise)))

            # Revenue generated
            revenue = round(daily_sales * actual_price, 2)

            records.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "product_id": p["product_id"],
                "product_name": p["product_name"],
                "category": p["category"],
                "price": actual_price,
                "promotion": promo,
                "sales": daily_sales,
                "revenue": revenue,
                "current_stock": p["current_stock"],
                "lead_time_days": p["lead_time_days"],
                "unit_cost": p["unit_cost"],
                "holding_cost": p["holding_cost"],
                "ordering_cost": p["ordering_cost"]
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

    # Save to CSV if path specified
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Generated synthetic sales dataset with {len(df)} records across {len(products_catalog)} products.")
        print(f"Source catalog: {catalog_source}")
        print(f"Saved dataset to: {output_file}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate retail sales dataset from product catalog CSV.")
    parser.add_argument("--catalog", type=str, default=DEFAULT_CATALOG_FILE, help="Path to products catalog CSV file")
    parser.add_argument("--output", type=str, default="retail_inventory_data.csv", help="Path to destination sales CSV")
    parser.add_argument("--days", type=int, default=730, help="Number of historical days to generate")
    parser.add_argument("--start-date", type=str, default="2024-01-01", help="Start date (YYYY-MM-DD)")

    args = parser.parse_args()
    generate_sales_data(catalog_source=args.catalog, output_file=args.output, start_date=args.start_date, days=args.days)
