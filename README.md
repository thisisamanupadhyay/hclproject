# 📦 Optimised Retail Inventory Management System

An end-to-end AI/ML retail inventory optimization system built with **Python**, **scikit-learn**, and **Streamlit**.

The system forecasts future product sales across retail categories and translates predictions into actionable supply-chain parameters: **Safety Stock**, **Reorder Points (ROP)**, and **Economic Order Quantity (EOQ)** to avoid costly stockouts and overstock.

---

## 🚀 Key Features

1. **CSV-Driven Product Catalog & Data Generator (`data_generator.py`)**:
   - **No manual code entry**: Product catalog parameters are loaded directly from `products_catalog.csv` (or any uploaded CSV).
   - Generates realistic daily sales transaction records (seasonality, holiday spikes, promotions, demand variance).
   - Embeds operational parameters: unit cost, holding cost, ordering cost, lead time, and current stock.

2. **Custom CSV Upload in Web Dashboard (`app.py`)**:
   - **Upload any CSV**: Instantly upload your own **Product Catalog CSV** or **Sales Transactions CSV** right from the sidebar.
   - **Auto-Detection**: Automatically identifies catalog files vs historical transaction logs.
   - **Instant Results**: Forecasts and reorder recommendations update dynamically in real time.
   - **Template Downloads**: 1-click download of ready-to-use CSV templates.

3. **Machine Learning Demand Forecasting (`model.py`)**:
   - **Model**: `RandomForestRegressor` with fixed seed for full reproducibility.
   - **Feature Engineering**: `day_of_week`, `month`, `price`, `promotion`, `lag_7`, `lag_14`, and leakage-free rolling averages (`rolling_mean_7`, `rolling_mean_30`).
   - **Evaluation**: Time-based train/test split evaluated on **MAE** and **RMSE** against a Moving Average baseline.
   - **Multi-step Forward Forecasting**: Dynamic recursive 30-day projection forward in time.

4. **Inventory Optimization Engine (`inventory_logic.py`)**:
   - **Safety Stock ($SS$)**: $SS = Z \times \sqrt{L} \times \sigma_d$ ($Z = 1.65$ for 95% cycle service level).
   - **Reorder Point ($ROP$)**: $ROP = (\bar{d} \times L) + SS$.
   - **Economic Order Quantity ($EOQ$)**: $EOQ = \sqrt{\frac{2 \times D \times S}{H}}$ (Wilson-Ford formula).
   - **Replenishment Decision**: Flags items as **Reorder Now**, **Healthy**, or **Overstock**.
   - **ABC Pareto Analysis**: Categorizes SKUs into Tier A (~70% revenue), Tier B (~20% revenue), and Tier C (~10% revenue).

5. **Streamlit Executive Dashboard (`app.py`)**:
   - **Sidebar Data Manager**: Switch between default CSV catalog and custom CSV upload with 1-click reload and template downloads.
   - **Tab 1: Overview & ABC Analysis**: High-level KPIs, interactive dual-axis Pareto chart, revenue share pie chart, and parsed CSV catalog preview.
   - **Tab 2: Demand Forecast**: Interactive product selector displaying actual vs predicted historical fit and next 30 days forecast.
   - **Tab 3: Reorder Recommendations**: Filterable table with colour-coded alerts (🔴 Reorder Now, 🟡 Overstock, 🟢 Healthy) and one-click CSV export.

---

## 📁 Project Structure

```text
hclproject/
│── products_catalog.csv     # Product operational parameters catalog (CSV data source)
│── data_generator.py        # Reads CSV catalog & generates realistic daily sales data
│── retail_inventory_data.csv# Generated 2-year daily sales transactions dataset
│── model.py                 # Feature engineering, RF model training, benchmarking & inference
│── inventory_logic.py       # Computes Safety Stock, ROP, EOQ, ABC tiers & reorder alerts
│── app.py                   # Streamlit multi-tab interactive dashboard with CSV upload
│── requirements.txt         # Python dependency specifications
└── README.md                # Documentation and execution guide
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard Directly
```bash
streamlit run app.py
```

---

## 📋 Catalog CSV Format (`products_catalog.csv`)

To add, edit, or swap products without touching code, simply edit `products_catalog.csv` or upload a CSV with these columns:

| Column | Type | Example | Description |
| :--- | :--- | :--- | :--- |
| `product_id` | Text | `P01` | Unique SKU identifier |
| `product_name` | Text | `Wireless Mouse` | Human-readable product name |
| `category` | Text | `Electronics` | Category grouping |
| `base_price` | Float | `25.0` | Standard retail price |
| `unit_cost` | Float | `14.0` | Cost of goods sold (COGS) per unit |
| `lead_time_days`| Integer | `5` | Supplier delivery lead time in days |
| `holding_cost` | Float | `2.50` | Annual storage/holding cost per unit |
| `ordering_cost`| Float | `45.0` | Fixed administrative cost per order |
| `base_demand` | Integer | `38` | Average daily customer demand |
| `current_stock`| Integer | `140` | Current on-hand inventory count |

*(Note: Column names are case-insensitive, and aliases like `sku`, `price`, `stock`, `cost` are also supported automatically!)*

---

## 🧪 Running Components Standalone

You can run each pipeline stage independently from the terminal:

- **Generate Data from CSV Catalog**:
  ```bash
  python data_generator.py --catalog products_catalog.csv --output retail_inventory_data.csv --days 730
  ```
- **Train & Benchmark ML Model**:
  ```bash
  python model.py
  ```
- **Test Inventory Logic**:
  ```bash
  python inventory_logic.py
  ```

---

## 📊 Inventory Logic Explained

| Metric | Formula | Purpose |
| :--- | :--- | :--- |
| **Safety Stock ($SS$)** | $Z \times \sqrt{L} \times \sigma_d$ | Buffers against supplier delays and unexpected demand spikes. |
| **Reorder Point ($ROP$)** | $(\bar{d} \times L) + SS$ | The exact inventory threshold that triggers purchase order creation. |
| **EOQ** | $\sqrt{\frac{2 D S}{H}}$ | The optimal order batch size that minimizes holding + ordering costs. |
| **ABC Analysis** | Pareto 70-20-10 on Annual Revenue | Identifies high-value items requiring tight continuous control. |
| **Status** | Current Stock vs ROP & EOQ | Classifies SKU health into `Reorder Now`, `Healthy`, or `Overstock`. |

---

## 🛡️ Design Principles
- **CSV-Centric**: Easily swap, edit, or upload product catalogs without editing code.
- **Self-contained**: Zero external internet, API keys, or databases required.
- **Fast Execution**: Fast feature computation and sub-second model training.
- **Reproducible**: Fixed random seeds guarantee deterministic data and training results.
