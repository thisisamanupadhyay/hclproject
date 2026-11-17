"""
app.py - Optimised Retail Inventory Management System Dashboard

Streamlit web application providing:
1. Flexible CSV Data Ingestion:
   - Automatically reads product catalog from `products_catalog.csv` (no manual code entry)
   - Supports uploading custom CSV files directly from the UI (either a Products Catalog or Sales History)
   - Downloadable CSV templates for quick testing and onboarding
2. Executive Overview: Key Performance Indicators (KPIs) and interactive ABC Pareto analysis
3. Demand Forecasting: Product-level historical vs predicted fit and 30-day forward forecasts
4. Reorder Recommendations: Actionable inventory matrix with status alerts and CSV export
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data_generator import (
    load_products_catalog,
    generate_sales_data,
    DEFAULT_CATALOG_FILE
)
from model import train_demand_model, forecast_next_days
from inventory_logic import generate_inventory_recommendations

# Set Page Config
st.set_page_config(
    page_title="Optimised Retail Inventory Management System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished dashboard look
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .badge-reorder {
        background-color: #ff4b4b;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-overstock {
        background-color: #ffa421;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-healthy {
        background-color: #21c354;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .source-banner {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_default_sales_data():
    """Loads default retail sales data, generating it from products_catalog.csv if needed."""
    sales_csv = "retail_inventory_data.csv"
    if not os.path.exists(DEFAULT_CATALOG_FILE):
        from data_generator import create_default_catalog_file
        create_default_catalog_file(DEFAULT_CATALOG_FILE)

    if not os.path.exists(sales_csv):
        with st.spinner("Generating retail dataset from products_catalog.csv..."):
            generate_sales_data(catalog_source=DEFAULT_CATALOG_FILE, output_file=sales_csv)

    df = pd.read_csv(sales_csv)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_resource
def load_default_model():
    """Loads or trains default demand forecasting model on retail_inventory_data.csv."""
    model_file = "demand_model.joblib"
    if not os.path.exists(model_file):
        with st.spinner("Training Random Forest demand forecasting model..."):
            return train_demand_model(save_path=model_file)
    import joblib
    return joblib.load(model_file)


def process_sales_dataframe(df_sales, is_custom=False):
    """Computes ML model forecast and inventory recommendations for a given sales dataframe."""
    # Ensure date column is datetime
    df_sales["date"] = pd.to_datetime(df_sales["date"])

    # Ensure required columns
    cols = df_sales.columns
    if "product_name" not in cols:
        df_sales["product_name"] = df_sales["product_id"].astype(str)
    if "category" not in cols:
        df_sales["category"] = "General"
    if "price" not in cols:
        df_sales["price"] = 25.0
    if "current_stock" not in cols:
        df_sales["current_stock"] = 100
    if "lead_time_days" not in cols:
        df_sales["lead_time_days"] = 7
    if "unit_cost" not in cols:
        df_sales["unit_cost"] = df_sales["price"] * 0.55
    if "holding_cost" not in cols:
        df_sales["holding_cost"] = (df_sales["price"] * 0.10).round(2)
    if "ordering_cost" not in cols:
        df_sales["ordering_cost"] = 45.0

    # Train model (or load default if standard)
    if not is_custom and os.path.exists("demand_model.joblib"):
        import joblib
        model_payload = joblib.load("demand_model.joblib")
    else:
        with st.spinner("Training forecasting model on the dataset..."):
            model_payload = train_demand_model(data_source=df_sales, save_path=None)

    # 30-Day Forward Forecast
    with st.spinner("Forecasting future 30-day demand..."):
        future_forecast_df = forecast_next_days(df_sales, model_payload, forecast_horizon_days=30)

    # Extract latest product parameters
    latest_products = df_sales.sort_values("date").groupby("product_id").last().reset_index()
    product_cols = [
        "product_id", "product_name", "category", "price",
        "current_stock", "lead_time_days", "unit_cost",
        "holding_cost", "ordering_cost"
    ]
    products_catalog = latest_products[[c for c in product_cols if c in latest_products.columns]].copy()

    # Generate inventory recommendations & ABC classification
    rec_df = generate_inventory_recommendations(products_catalog, future_forecast_df)

    return df_sales, model_payload, future_forecast_df, rec_df


# --- SIDEBAR CONFIGURATION ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2897/2897818.png", width=60)
st.sidebar.title("Inventory System")
st.sidebar.caption("AI-Powered Retail Demand & Stock Optimization")
st.sidebar.markdown("---")

# 1. Data Source Selection
st.sidebar.subheader("📂 Data Source")
data_mode = st.sidebar.radio(
    "Select Data Input Mode:",
    ["📄 Default Catalog (`products_catalog.csv`)", "📤 Upload Custom CSV File"],
    index=0
)

active_source_label = "Default Catalog (products_catalog.csv)"
is_custom_data = False

if data_mode == "📄 Default Catalog (`products_catalog.csv`)":
    if st.sidebar.button("🔄 Reload from CSV & Regenerate", help="Re-reads products_catalog.csv from disk and re-runs pipeline"):
        st.cache_data.clear()
        st.cache_resource.clear()
        # Regenerate sales dataset from products_catalog.csv
        generate_sales_data(catalog_source=DEFAULT_CATALOG_FILE, output_file="retail_inventory_data.csv")
        train_demand_model(data_source="retail_inventory_data.csv", save_path="demand_model.joblib")
        st.sidebar.success("Catalog and sales data refreshed from disk!")
        st.rerun()

    df_sales_raw = load_default_sales_data()
    model_payload = load_default_model()
    # Compute recommendations
    from inventory_logic import generate_inventory_recommendations
    latest_products = df_sales_raw.sort_values("date").groupby("product_id").last().reset_index()
    product_cols = [
        "product_id", "product_name", "category", "price",
        "current_stock", "lead_time_days", "unit_cost",
        "holding_cost", "ordering_cost"
    ]
    catalog_summary = latest_products[product_cols].copy()
    future_forecast_df = forecast_next_days(df_sales_raw, model_payload, forecast_horizon_days=30)
    rec_df = generate_inventory_recommendations(catalog_summary, future_forecast_df)
    df_sales = df_sales_raw

else:
    # Custom CSV upload mode
    is_custom_data = True
    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        help="Upload either a Products Catalog CSV or a Historical Sales Transactions CSV."
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            cols_lower = [str(c).strip().lower() for c in uploaded_df.columns]

            if "date" in cols_lower and "sales" in cols_lower:
                # Historical Sales CSV
                st.sidebar.success(f"✅ Loaded Sales CSV: **{uploaded_file.name}** ({len(uploaded_df)} records)")
                active_source_label = f"Uploaded Sales: {uploaded_file.name}"
                df_sales, model_payload, future_forecast_df, rec_df = process_sales_dataframe(uploaded_df, is_custom=True)
            else:
                # Products Catalog CSV
                st.sidebar.success(f"✅ Loaded Products Catalog: **{uploaded_file.name}** ({len(uploaded_df)} products)")
                active_source_label = f"Uploaded Catalog: {uploaded_file.name}"
                with st.spinner("Simulating sales transactions from uploaded catalog..."):
                    gen_df = generate_sales_data(catalog_source=uploaded_df, output_file=None, days=730)
                df_sales, model_payload, future_forecast_df, rec_df = process_sales_dataframe(gen_df, is_custom=True)

        except Exception as e:
            st.sidebar.error(f"Error parsing uploaded CSV: {str(e)}")
            st.info("Falling back to default catalog dataset.")
            df_sales = load_default_sales_data()
            model_payload = load_default_model()
            future_forecast_df = forecast_next_days(df_sales, model_payload, forecast_horizon_days=30)
            latest_products = df_sales.sort_values("date").groupby("product_id").last().reset_index()
            catalog_summary = latest_products[[c for c in ["product_id", "product_name", "category", "price", "current_stock", "lead_time_days", "unit_cost", "holding_cost", "ordering_cost"] if c in latest_products.columns]].copy()
            rec_df = generate_inventory_recommendations(catalog_summary, future_forecast_df)
    else:
        st.sidebar.warning("⚠️ No file uploaded yet. Showing default catalog.")
        df_sales = load_default_sales_data()
        model_payload = load_default_model()
        future_forecast_df = forecast_next_days(df_sales, model_payload, forecast_horizon_days=30)
        latest_products = df_sales.sort_values("date").groupby("product_id").last().reset_index()
        catalog_summary = latest_products[[c for c in ["product_id", "product_name", "category", "price", "current_stock", "lead_time_days", "unit_cost", "holding_cost", "ordering_cost"] if c in latest_products.columns]].copy()
        rec_df = generate_inventory_recommendations(catalog_summary, future_forecast_df)

# CSV Templates Download Section in Sidebar
st.sidebar.markdown("---")
with st.sidebar.expander("📥 Download CSV Templates"):
    st.caption("Use these templates to prepare your own data:")

    # 1. Product Catalog Template
    if os.path.exists(DEFAULT_CATALOG_FILE):
        with open(DEFAULT_CATALOG_FILE, "r") as f:
            cat_template_csv = f.read()
    else:
        cat_template_csv = "product_id,product_name,category,base_price,unit_cost,lead_time_days,holding_cost,ordering_cost,base_demand,current_stock\nP01,Sample Product,Electronics,25.0,14.0,5,2.5,45.0,30,120\n"

    st.download_button(
        label="📄 Products Catalog Template",
        data=cat_template_csv,
        file_name="products_catalog_template.csv",
        mime="text/csv",
        use_container_width=True
    )

    # 2. Sales Transactions Template (Sample 50 rows)
    sample_sales_csv = df_sales.head(50).to_csv(index=False)
    st.download_button(
        label="📊 Sales History Template",
        data=sample_sales_csv,
        file_name="sales_history_template.csv",
        mime="text/csv",
        use_container_width=True
    )

st.sidebar.markdown("---")
total_products = len(rec_df)
reorder_count = len(rec_df[rec_df["status"] == "Reorder Now"])
overstock_count = len(rec_df[rec_df["status"] == "Overstock"])
healthy_count = len(rec_df[rec_df["status"] == "Healthy"])

st.sidebar.metric("Total Products", total_products)
st.sidebar.metric("🚨 Need Reorder", reorder_count)
st.sidebar.metric("⚠️ Overstocked", overstock_count)
st.sidebar.metric("✅ Healthy Stock", healthy_count)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Quick Tips:**\n"
    "- **Safety Stock (SS)** buffers lead-time volatility.\n"
    "- **Reorder Point (ROP)** triggers purchase replenishment.\n"
    "- **EOQ** minimizes total ordering & holding costs."
)

# --- HEADER SECTION ---
st.title("📦 Optimised Retail Inventory Management System")
st.markdown("Automated demand forecasting, safety stock determination, and inventory replenishment recommendations from CSV data.")

# Active Data Banner
col_b1, col_b2 = st.columns([3, 1])
with col_b1:
    st.info(f"📁 **Active Data Source:** `{active_source_label}` | **SKUs Tracked:** {total_products}")
with col_b2:
    if is_custom_data:
        st.caption("🟢 Using uploaded custom data")
    else:
        st.caption("🔵 Using `products_catalog.csv`")

# Tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Overview & ABC Analysis",
    "📈 Demand Forecast",
    "📋 Reorder Recommendations"
])

# ==========================================
# TAB 1: OVERVIEW & ABC ANALYSIS
# ==========================================
with tab1:
    st.subheader("Executive Inventory Health Dashboard")

    metrics_meta = model_payload.get("metrics", {})
    ml_mae = metrics_meta.get("ml_mae", 3.2)
    improvement = metrics_meta.get("mae_improvement_pct", 44.8)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Total Monitored SKUs",
            value=f"{total_products} items",
            delta=f"{total_products} Active Products"
        )
    with col2:
        st.metric(
            label="Urgent Reorder Required",
            value=f"{reorder_count} SKUs",
            delta=f"-{reorder_count} Stock Risk",
            delta_color="inverse"
        )
    with col3:
        st.metric(
            label="Overstocked Items",
            value=f"{overstock_count} SKUs",
            delta=f"{overstock_count} Excess Capital",
            delta_color="off"
        )
    with col4:
        st.metric(
            label="ML Forecast Accuracy (MAE)",
            value=f"{ml_mae:.2f} units",
            delta=f"{improvement:.1f}% vs Moving Avg",
            delta_color="normal"
        )

    st.markdown("---")

    # ABC Pareto Analysis
    st.subheader("ABC Revenue Analysis (Pareto Classification)")
    st.caption("Class A items generate ~70% of revenue, Class B ~20%, and Class C ~10%. Prioritize Class A replenishment.")

    c1, c2 = st.columns([2, 1])

    with c1:
        # Pareto Chart: Revenue per product + Cumulative percentage line
        abc_sorted = rec_df.sort_values(by="annual_revenue", ascending=False).copy()

        fig_abc = go.Figure()
        fig_abc.add_trace(go.Bar(
            x=abc_sorted["product_name"],
            y=abc_sorted["annual_revenue"],
            name="Annual Revenue ($)",
            marker=dict(
                color=abc_sorted["abc_class"].map({
                    "A": "#1f77b4",
                    "B": "#ff7f0e",
                    "C": "#2ca02c"
                }).fillna("#1f77b4")
            ),
            hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.2f}<extra></extra>"
        ))
        fig_abc.add_trace(go.Scatter(
            x=abc_sorted["product_name"],
            y=abc_sorted["cum_revenue_pct"],
            name="Cumulative Revenue %",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#d62728", width=2.5),
            hovertemplate="Cumulative: %{y:.1f}%<extra></extra>"
        ))

        fig_abc.update_layout(
            title="Revenue Contribution and Cumulative Curve",
            xaxis=dict(title="Products", tickangle=-45),
            yaxis=dict(title="Annual Projected Revenue ($)"),
            yaxis2=dict(
                title="Cumulative Revenue %",
                overlaying="y",
                side="right",
                range=[0, 105],
                showgrid=False
            ),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
            margin=dict(t=40, b=100, l=40, r=40),
            height=420
        )
        st.plotly_chart(fig_abc, use_container_width=True)

    with c2:
        # Category breakdown pie & stats
        abc_summary = rec_df.groupby("abc_class").agg(
            SKUs=("product_id", "count"),
            Revenue=("annual_revenue", "sum")
        ).reset_index()
        total_rev = max(0.01, abc_summary["Revenue"].sum())
        abc_summary["Revenue %"] = (abc_summary["Revenue"] / total_rev) * 100

        fig_pie = px.pie(
            abc_summary,
            names="abc_class",
            values="Revenue",
            title="Revenue Share by ABC Tier",
            color="abc_class",
            color_discrete_map={"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"},
            hole=0.45
        )
        fig_pie.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)

        st.dataframe(
            abc_summary.rename(columns={"abc_class": "Tier"}),
            hide_index=True,
            column_config={
                "Revenue": st.column_config.NumberColumn(format="$%.2f"),
                "Revenue %": st.column_config.NumberColumn(format="%.1f%%")
            },
            use_container_width=True
        )

    # Catalog preview expander
    with st.expander("🔍 View Active Product Catalog Parameters (from CSV)"):
        st.caption("Operational parameters parsed directly from the CSV input (no manual coding required):")
        display_catalog = rec_df[[
            "product_id", "product_name", "category", "price", "unit_cost",
            "current_stock", "lead_time_days", "avg_daily_demand", "reorder_point", "eoq"
        ]].copy()
        st.dataframe(display_catalog, hide_index=True, use_container_width=True)


# ==========================================
# TAB 2: DEMAND FORECAST
# ==========================================
with tab2:
    st.subheader("Machine Learning Demand Forecasting")
    st.caption("Inspect historical test set model accuracy and inspect future 30-day forecast per product.")

    # Product selector
    product_options = {
        f"{row['product_id']} - {row['product_name']} ({row['category']})": row["product_id"]
        for _, row in rec_df.iterrows()
    }
    if len(product_options) > 0:
        selected_label = st.selectbox("Select Product to Inspect:", list(product_options.keys()))
        selected_pid = product_options[selected_label]

        # Selected product info
        prod_row = rec_df[rec_df["product_id"] == selected_pid].iloc[0]

        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        f_col1.metric("Category", prod_row["category"])
        f_col2.metric("Selling Price", f"${prod_row['price']:.2f}")
        f_col3.metric("Lead Time", f"{prod_row['lead_time_days']} days")
        f_col4.metric("Avg Daily Demand", f"{prod_row['avg_daily_demand']:.1f} units")

        # Chart 1: Actual vs Predicted on unseen test period
        test_eval_df = model_payload.get("test_df")
        if test_eval_df is not None and not test_eval_df.empty:
            prod_test = test_eval_df[test_eval_df["product_id"] == selected_pid].sort_values("date")
            if not prod_test.empty:
                fig_test = go.Figure()
                fig_test.add_trace(go.Scatter(
                    x=prod_test["date"],
                    y=prod_test["sales"],
                    mode="lines+markers",
                    name="Actual Sales",
                    line=dict(color="#1f77b4", width=2.5)
                ))
                fig_test.add_trace(go.Scatter(
                    x=prod_test["date"],
                    y=prod_test["predicted_sales"],
                    mode="lines",
                    name="Random Forest Forecast",
                    line=dict(color="#ff7f0e", width=2.5, dash="dash")
                ))
                fig_test.add_trace(go.Scatter(
                    x=prod_test["date"],
                    y=prod_test["rolling_mean_7"],
                    mode="lines",
                    name="7-Day Moving Avg Baseline",
                    line=dict(color="#7f7f7f", width=1.5, dash="dot")
                ))

                fig_test.update_layout(
                    title=f"Test Evaluation: Actual vs ML Prediction - {prod_row['product_name']}",
                    xaxis_title="Date",
                    yaxis_title="Daily Units Sold",
                    hovermode="x unified",
                    height=360,
                    margin=dict(t=40, b=20, l=40, r=40)
                )
                st.plotly_chart(fig_test, use_container_width=True)

        # Chart 2: Next 30 Days Forecast
        st.markdown("#### 📅 Next 30 Days Forward Forecast")
        prod_future = future_forecast_df[future_forecast_df["product_id"] == selected_pid].sort_values("date")

        if not prod_future.empty:
            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(
                x=prod_future["date"],
                y=prod_future["predicted_sales"],
                mode="lines+markers",
                name="Projected Daily Sales",
                line=dict(color="#2ca02c", width=3),
                marker=dict(size=6)
            ))
            # Rolling 7-day trend line
            prod_future["rolling_pred"] = prod_future["predicted_sales"].rolling(7, min_periods=1).mean()
            fig_future.add_trace(go.Scatter(
                x=prod_future["date"],
                y=prod_future["rolling_pred"],
                mode="lines",
                name="7-Day Trend",
                line=dict(color="#17becf", width=2, dash="dash")
            ))

            fig_future.update_layout(
                title=f"Next 30 Days Forward Forecast - {prod_row['product_name']}",
                xaxis_title="Future Date",
                yaxis_title="Predicted Daily Units Sold",
                hovermode="x unified",
                height=340,
                margin=dict(t=40, b=20, l=40, r=40)
            )
            st.plotly_chart(fig_future, use_container_width=True)
    else:
        st.warning("No products available to forecast.")


# ==========================================
# TAB 3: REORDER RECOMMENDATIONS
# ==========================================
with tab3:
    st.subheader("Inventory Optimization & Reorder Recommendations")
    st.caption("Actionable matrix computing Safety Stock, Reorder Point (ROP), EOQ, and replenishments.")

    # Filter row
    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        cat_filter = st.selectbox("Filter Category:", ["All"] + sorted(list(rec_df["category"].dropna().unique())))
    with f2:
        status_filter = st.selectbox("Filter Status:", ["All", "Reorder Now", "Healthy", "Overstock"])

    display_df = rec_df.copy()
    if cat_filter != "All":
        display_df = display_df[display_df["category"] == cat_filter]
    if status_filter != "All":
        display_df = display_df[display_df["status"] == status_filter]

    # Reorder table styling
    def format_status(val):
        if val == "Reorder Now":
            return "color: #b30000; font-weight: bold; background-color: #ffd6d6"
        elif val == "Overstock":
            return "color: #8a6d00; font-weight: bold; background-color: #fff2cc"
        return "color: #0d6832; font-weight: bold; background-color: #d4edda"

    # Display columns
    table_columns = [
        "product_id", "product_name", "category", "price",
        "current_stock", "lead_time_days", "safety_stock",
        "reorder_point", "eoq", "days_of_supply",
        "recommended_order", "status", "abc_class"
    ]

    styled_df = display_df[table_columns].style.map(format_status, subset=["status"])

    st.dataframe(
        styled_df,
        hide_index=True,
        column_config={
            "product_id": "SKU",
            "product_name": "Product Name",
            "category": "Category",
            "price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "current_stock": st.column_config.NumberColumn("Current Stock", format="%d"),
            "lead_time_days": st.column_config.NumberColumn("Lead Time (d)", format="%d"),
            "safety_stock": st.column_config.NumberColumn("Safety Stock", format="%d"),
            "reorder_point": st.column_config.NumberColumn("ROP", format="%d"),
            "eoq": st.column_config.NumberColumn("EOQ", format="%d"),
            "days_of_supply": st.column_config.NumberColumn("Supply (days)", format="%.1f"),
            "recommended_order": st.column_config.NumberColumn("Order Qty", format="%d"),
            "status": "Stock Status",
            "abc_class": "ABC Tier"
        },
        use_container_width=True
    )

    st.markdown("---")

    # Actionable download section
    down_col1, down_col2 = st.columns([3, 1])
    with down_col1:
        st.write(
            f"Showing **{len(display_df)}** items. "
            f"Total recommended reorder quantity: **{int(display_df['recommended_order'].sum()):,} units**."
        )
    with down_col2:
        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV Report",
            data=csv_data,
            file_name="inventory_reorder_recommendations.csv",
            mime="text/csv",
            use_container_width=True
        )
