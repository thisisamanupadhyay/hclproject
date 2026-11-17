"""
inventory_logic.py - Retail Inventory Optimization & ABC Classification

Computes standard supply-chain metrics based on ML forecasted demand:
1. Safety Stock (SS) via statistical service-level formula: SS = Z * sqrt(L) * sigma_d
2. Reorder Point (ROP): ROP = (avg_demand * L) + SS
3. Economic Order Quantity (EOQ): EOQ = sqrt((2 * D * S) / H)
4. Recommended Order Quantity: Actionable quantity to reorder when stock is below ROP
5. Inventory Health Status: 'Reorder Now', 'Healthy', or 'Overstock'
6. ABC Classification: Revenue-based Pareto categorization (A: Top ~70%, B: Next ~20%, C: Bottom ~10%)
"""

import numpy as np
import pandas as pd


def compute_safety_stock(lead_time_days: float, std_daily_demand: float, service_level_z: float = 1.65) -> int:
    """
    Computes Safety Stock required to buffer demand uncertainty during lead time.
    Formula: SS = Z * sqrt(Lead Time) * std(Daily Demand)
    Z = 1.65 corresponds to a 95% cycle service level.
    """
    ss = service_level_z * np.sqrt(max(1.0, lead_time_days)) * max(1.0, std_daily_demand)
    return int(np.ceil(ss))


def compute_reorder_point(lead_time_days: float, avg_daily_demand: float, safety_stock: int) -> int:
    """
    Computes the Reorder Point (ROP).
    When current inventory drops to or below this level, a new order must be placed.
    Formula: ROP = (Lead Time * Average Daily Demand) + Safety Stock
    """
    rop = (lead_time_days * avg_daily_demand) + safety_stock
    return int(np.ceil(rop))


def compute_eoq(annual_demand: float, ordering_cost: float, holding_cost: float) -> int:
    """
    Computes Economic Order Quantity (EOQ) using the Wilson-Ford formula.
    Balances annual fixed ordering costs against inventory holding costs.
    Formula: EOQ = sqrt((2 * D * S) / H)
    """
    if holding_cost <= 0 or annual_demand <= 0 or ordering_cost <= 0:
        return int(max(1, annual_demand / 12))
    eoq = np.sqrt((2.0 * annual_demand * ordering_cost) / holding_cost)
    return int(np.ceil(eoq))


def perform_abc_analysis(df_inventory: pd.DataFrame) -> pd.DataFrame:
    """
    Performs ABC Analysis on products based on their expected annual sales revenue.
    - Class A: Top ~70% of total revenue (high priority, tight control)
    - Class B: Next ~20% of revenue (medium priority)
    - Class C: Remaining ~10% of revenue (lower priority, loose control)
    """
    df = df_inventory.copy()
    df = df.sort_values(by="annual_revenue", ascending=False).reset_index(drop=True)

    total_revenue = df["annual_revenue"].sum()
    if total_revenue <= 0:
        df["abc_class"] = "B"
        df["cum_revenue_pct"] = 100.0
        return df

    # Calculate cumulative percentage of revenue
    df["cum_revenue"] = df["annual_revenue"].cumsum()
    df["cum_revenue_pct"] = (df["cum_revenue"] / total_revenue) * 100.0

    # Assign ABC classification
    # Using shift to evaluate prior cumulative percentage so threshold items are captured cleanly
    prev_cum = df["cum_revenue_pct"].shift(1).fillna(0)
    
    classes = []
    for p in prev_cum:
        if p < 70.0:
            classes.append("A")
        elif p < 90.0:
            classes.append("B")
        else:
            classes.append("C")

    df["abc_class"] = classes
    return df


def generate_inventory_recommendations(products_df: pd.DataFrame,
                                       forecast_df: pd.DataFrame,
                                       service_level_z: float = 1.65) -> pd.DataFrame:
    """
    Combines operational product metadata with forecasted demand to produce
    comprehensive inventory optimization recommendations.

    Parameters:
        products_df: DataFrame with product attributes (id, name, category, price, current_stock,
                     lead_time_days, unit_cost, holding_cost, ordering_cost)
        forecast_df: Daily forecasted sales (product_id, date, predicted_sales)
        service_level_z: Z-score for service level (default 1.65 = 95%)

    Returns:
        pd.DataFrame: Table of recommendations and inventory status per product.
    """
    records = []

    # Calculate forecast statistics per product
    forecast_stats = forecast_df.groupby("product_id")["predicted_sales"].agg(
        avg_demand="mean",
        std_demand="std"
    ).reset_index()

    # Merge product parameters with forecast statistics
    merged = pd.merge(products_df, forecast_stats, on="product_id", how="left")

    for _, row in merged.iterrows():
        pid = str(row["product_id"])
        pname = str(row.get("product_name", pid))
        cat = str(row.get("category", "General"))
        price = float(row["price"]) if ("price" in row and pd.notna(row["price"])) else float(row.get("base_price", 25.0))
        stock = int(row["current_stock"]) if ("current_stock" in row and pd.notna(row["current_stock"])) else 100
        lead_time = float(row["lead_time_days"]) if ("lead_time_days" in row and pd.notna(row["lead_time_days"])) else 7.0
        unit_cost = float(row["unit_cost"]) if ("unit_cost" in row and pd.notna(row["unit_cost"])) else round(price * 0.55, 2)
        holding_cost = float(row["holding_cost"]) if ("holding_cost" in row and pd.notna(row["holding_cost"])) else max(0.5, round(price * 0.10, 2))
        ordering_cost = float(row["ordering_cost"]) if ("ordering_cost" in row and pd.notna(row["ordering_cost"])) else 45.0

        avg_daily = float(row["avg_demand"]) if pd.notna(row["avg_demand"]) else 20.0
        # If std is NaN or 0, fallback to 20% of mean demand
        std_daily = float(row["std_demand"]) if (pd.notna(row["std_demand"]) and row["std_demand"] > 0) else (avg_daily * 0.20)

        # 1. Annual Demand
        annual_demand = avg_daily * 365.0
        annual_revenue = annual_demand * price

        # 2. Safety Stock
        safety_stock = compute_safety_stock(lead_time, std_daily, service_level_z)

        # 3. Reorder Point (ROP)
        rop = compute_reorder_point(lead_time, avg_daily, safety_stock)

        # 4. Economic Order Quantity (EOQ)
        eoq = compute_eoq(annual_demand, ordering_cost, holding_cost)

        # 5. Overstock Upper Bound (e.g. ROP + 1.5 * EOQ)
        overstock_threshold = rop + int(1.5 * eoq)

        # 6. Inventory Status & Recommended Order Quantity
        if stock <= rop:
            status = "Reorder Now"
            # Reorder at least EOQ or enough to return to (ROP + EOQ)
            recommended_order = max(eoq, (rop - stock + eoq))
        elif stock > overstock_threshold:
            status = "Overstock"
            recommended_order = 0
        else:
            status = "Healthy"
            recommended_order = 0

        # Days of Supply remaining based on average daily demand
        days_of_supply = round(stock / max(0.1, avg_daily), 1)

        records.append({
            "product_id": pid,
            "product_name": pname,
            "category": cat,
            "price": round(price, 2),
            "unit_cost": round(unit_cost, 2),
            "current_stock": stock,
            "avg_daily_demand": round(avg_daily, 2),
            "std_daily_demand": round(std_daily, 2),
            "lead_time_days": int(lead_time),
            "safety_stock": safety_stock,
            "reorder_point": rop,
            "eoq": eoq,
            "days_of_supply": days_of_supply,
            "status": status,
            "recommended_order": recommended_order,
            "annual_revenue": round(annual_revenue, 2)
        })

    result_df = pd.DataFrame(records)
    # Apply ABC classification
    result_df = perform_abc_analysis(result_df)
    return result_df


if __name__ == "__main__":
    from data_generator import generate_sales_data, get_products_catalog
    print("Testing inventory logic independently...")
    df_cat = get_products_catalog()
    # Dummy forecast for testing
    dummy_forecast = []
    for pid in df_cat["product_id"]:
        for d in range(30):
            dummy_forecast.append({
                "product_id": pid,
                "date": f"2026-01-{d+1:02d}",
                "predicted_sales": np.random.randint(15, 60)
            })
    df_fc = pd.DataFrame(dummy_forecast)
    rec = generate_inventory_recommendations(df_cat, df_fc)
    print(rec[["product_id", "product_name", "current_stock", "reorder_point", "eoq", "status", "abc_class"]].head())
