"""
Sales Dashboard App — Streamlit (single-file)
Author: ChatGPT

Features
- **User login** (username/password) via `streamlit-authenticator`
- Upload a CSV of transactional sales data (or use demo data)
- Auto-clean columns & infer dtypes (dates, currency)
- Global filters: Date range, Region, Channel, Category, Product
- KPI cards: Revenue, Profit, Orders, AOV, Units, Margin %
- Charts (Plotly):
  * Revenue over time (auto-granularity: day/week/month)
  * Sales by Category / Subcategory
  * Top Products (Pareto)
  * Channel mix
  * Region / Country performance (table)
- Download filtered dataset as CSV

How to run
1) Install deps:  
   pip install -r requirements.txt  
   (or) pip install streamlit pandas numpy plotly python-dateutil pycountry
2) Start:  
   streamlit run app.py

Deployment (multi‑user)
- Add `streamlit-authenticator` to requirements.
- Set credentials in **Streamlit Secrets** or environment variables (see AUTH section below).
- Deploy on Streamlit Community Cloud, Render, Railway, or Fly.io.

Example `requirements.txt`:
```
streamlit
streamlit-authenticator
pandas
numpy
plotly
python-dateutil
pycountry
```


Acceptable column names (case-insensitive; underscores/hyphens/spaces ignored)
- order date: [orderdate, date, order_date, invoice_date, txn_date]
- order id: [orderid, order_id, invoice, invoice_no, orderno]
- customer id: [customerid, customer_id, cust, account]
- product: [product, sku, item]
- category: [category]
- subcategory: [subcategory, sub_category]
- channel: [channel, source]
- region: [region, state, province]
- country: [country]
- quantity: [qty, quantity, units]
- unit price: [price, unitprice, unit_price]
- discount: [discount, promo]
- cost: [cost, cogs]
- sales/revenue: [sales, revenue, amount, total, line_total]

If sales is missing, computed as quantity * unit price * (1 - discount).
Profit = sales - (cost if provided else 0).
"""

from __future__ import annotations
import io
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from dateutil.relativedelta import relativedelta
import streamlit_authenticator as stauth

# -----------------------------
# -------- AUTHENTICATION ------
# -----------------------------

# Expected secrets structure (Streamlit: Settings → Secrets):
# [[auth.credentials.usernames]]
# haris.name = "Muhammad Haris"
# haris.password = "hashed_password_here"
# haris.email = "haris@example.com"
#
# To generate hashes locally:
#   python -c "import streamlit_authenticator as stauth; print(stauth.Hasher(['YourPass']).generate())"

AUTH_CONFIG = {
    "cookie": {"name": "salesdash_auth", "key": "replace_with_random_key", "expiry_days": 30},
    "preauthorized": {"emails": []},
}

# Load credentials from secrets if present
if "auth" in st.secrets and "credentials" in st.secrets["auth"]:
    credentials = {"usernames": {}}
    for uname, u in st.secrets["auth"]["credentials"].items():
        credentials["usernames"][uname] = {
            "name": u.get("name", uname),
            "email": u.get("email", ""),
            "password": u.get("password"),  # must be hashed
        }
else:
    # Fallback demo credentials (username: demo / password: demo)
    demo_hash = stauth.Hasher(["demo"]).generate()[0]
    credentials = {
        "usernames": {
            "demo": {"name": "Demo User", "email": "demo@example.com", "password": demo_hash}
        }
    }

authenticator = stauth.Authenticate(
    credentials,
    AUTH_CONFIG["cookie"]["name"],
    AUTH_CONFIG["cookie"]["key"],
    AUTH_CONFIG["cookie"]["expiry_days"],
)

st.sidebar.title("🔐 Sign in")
name, authentication_status, username = authenticator.login("Login", "sidebar")

if not authentication_status:
    if authentication_status is False:
        st.sidebar.error("Username/password is incorrect")
    st.stop()

# Show logout and user info once authenticated
with st.sidebar:
    st.success(f"Signed in as {name}")
    authenticator.logout("Logout", "sidebar")

# -----------------------------
# ----------- UTILS -----------
# -----------------------------

st.set_page_config(page_title="Sales Dashboard", layout="wide")

@st.cache_data(show_spinner=False)
def load_demo_data(rows: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01")
    dates = start + pd.to_timedelta(rng.integers(0, 365, size=rows), unit="D")
    regions = ["Ontario", "Quebec", "BC", "Prairies", "Atlantic"]
    countries = ["Canada"] * rows
    channels = ["Online", "Retail", "Distributor"]
    categories = ["Electronics", "Home", "Beauty", "Sports"]
    subcats = {
        "Electronics": ["Phones", "Laptops", "Audio"],
        "Home": ["Kitchen", "Decor", "Cleaning"],
        "Beauty": ["Skincare", "Makeup", "Hair"],
        "Sports": ["Fitness", "Outdoor", "Apparel"],
    }
    cat_choice = rng.choice(categories, size=rows)
    subcat = [rng.choice(subcats[c]) for c in cat_choice]
    product_base = [
        "Aurora", "Nimbus", "Vertex", "Pulse", "Quanta", "Zenith",
        "Nova", "Echo", "Orbit", "Catalyst"
    ]
    product = [f"{p} {s}" for p, s in zip(rng.choice(product_base, size=rows), subcat)]
    qty = rng.integers(1, 6, size=rows)
    price = rng.integers(15, 600, size=rows).astype(float)
    discount = rng.choice([0.0, 0.05, 0.1, 0.15], size=rows, p=[0.6, 0.2, 0.15, 0.05])
    cost = price * rng.uniform(0.4, 0.8, size=rows)

    df = pd.DataFrame({
        "OrderDate": dates,
        "OrderID": rng.integers(100000, 999999, size=rows).astype(str),
        "CustomerID": rng.integers(1000, 9999, size=rows).astype(str),
        "Product": product,
        "Category": cat_choice,
        "Subcategory": subcat,
        "Channel": rng.choice(channels, size=rows, p=[0.55, 0.35, 0.10]),
        "Region": rng.choice(regions, size=rows),
        "Country": countries,
        "Quantity": qty,
        "UnitPrice": price,
        "Discount": discount,
        "Cost": cost,
    })
    df["Sales"] = (df["Quantity"] * df["UnitPrice"]) * (1 - df["Discount"])
    return df

CANON = {
    "order_date": ["orderdate", "date", "order_date", "invoice_date", "txn_date"],
    "order_id": ["orderid", "order_id", "invoice", "invoice_no", "orderno"],
    "customer_id": ["customerid", "customer_id", "cust", "account"],
    "product": ["product", "sku", "item"],
    "category": ["category"],
    "subcategory": ["subcategory", "sub_category"],
    "channel": ["channel", "source"],
    "region": ["region", "state", "province"],
    "country": ["country"],
    "quantity": ["qty", "quantity", "units"],
    "unit_price": ["price", "unitprice", "unit_price"],
    "discount": ["discount", "promo"],
    "cost": ["cost", "cogs"],
    "sales": ["sales", "revenue", "amount", "total", "line_total"],
}

NORMALIZE = lambda s: re.sub(r"[^a-z0-9]", "", str(s).strip().lower())


def canonicalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    colmap: Dict[str, str] = {}
    norm_to_orig = {NORMALIZE(c): c for c in df.columns}

    for canon, aliases in CANON.items():
        for a in aliases:
            if a in norm_to_orig:
                colmap[canon] = norm_to_orig[a]
                break
    # Rename in-place to canonical names (keep originals if not mapped)
    rename_map = {v: k for k, v in colmap.items()}
    out = df.rename(columns=rename_map).copy()
    return out, colmap


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    for c in ["quantity", "unit_price", "discount", "cost", "sales"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def compute_fields(df: pd.DataFrame) -> pd.DataFrame:
    # Compute Sales if missing
    if "sales" not in df.columns and {"quantity", "unit_price"}.issubset(df.columns):
        disc = df["discount"] if "discount" in df.columns else 0
        df["sales"] = (df["quantity"] * df["unit_price"]) * (1 - disc.fillna(0))
    # Profit
    if "cost" in df.columns:
        df["profit"] = df["sales"].fillna(0) - df["cost"].fillna(0)
    else:
        df["profit"] = np.nan
    # Orders proxy
    if "order_id" in df.columns:
        df["_orders"] = df["order_id"].astype(str)
    else:
        df["_orders"] = np.arange(len(df)).astype(str)  # fallback
    return df


def auto_granularity(start: pd.Timestamp, end: pd.Timestamp) -> str:
    days = (end - start).days if pd.notna(start) and pd.notna(end) else 0
    if days <= 62:
        return "D"  # daily
    elif days <= 370:
        return "W"  # weekly
    else:
        return "M"  # monthly


# -----------------------------
# ---------- SIDEBAR ----------
# -----------------------------

st.sidebar.title("📊 Sales Dashboard")
st.sidebar.caption("Upload your CSV or explore with demo data.")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"]) 
use_demo = st.sidebar.toggle("Use demo data", value=uploaded is None)

if use_demo:
    df_raw = load_demo_data()
else:
    if uploaded is None:
        st.stop()
    df_raw = pd.read_csv(uploaded)

# Prepare data
_df, colmap = canonicalize_columns(df_raw)
_df = coerce_types(_df)
_df = compute_fields(_df)

required_cols = ["order_date", "sales"]
if any(c not in _df.columns for c in required_cols):
    st.error("Your data must have at least an Order Date and either Sales or (Quantity & Unit Price). Please adjust your file.")
    st.write("Detected columns:", list(_df.columns))
    st.stop()

min_date = pd.to_datetime(_df["order_date"].min())
max_date = pd.to_datetime(_df["order_date"].max())

# Filters
st.sidebar.subheader("Filters")
start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)

multis = {}
for col in ["region", "country", "channel", "category", "product"]:
    if col in _df.columns:
        opts = sorted([x for x in _df[col].dropna().astype(str).unique()])
        if len(opts) > 0:
            multis[col] = st.sidebar.multiselect(col.capitalize(), options=opts, default=[])

# Apply filters
mask = (
    (_df["order_date"] >= pd.to_datetime(start_date)) &
    (_df["order_date"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
)
for col, sel in multis.items():
    if sel:
        mask &= _df[col].astype(str).isin(sel)

DF = _df.loc[mask].copy()

# -----------------------------
# ---------- HEADER -----------
# -----------------------------

st.title("Sales Performance Dashboard")
st.caption("Power BI–style quick insights, no setup headaches.")

# KPIs
rev = float(DF["sales"].sum())
orders = DF["_orders"].nunique() if len(DF) else 0
units = int(DF.get("quantity", pd.Series(dtype=float)).sum()) if "quantity" in DF.columns else np.nan
profit = float(DF["profit"].sum()) if DF["profit"].notna().any() else np.nan

aov = rev / orders if orders else np.nan
margin = profit / rev if (rev and not np.isnan(profit)) else np.nan

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Revenue", f"${rev:,.0f}")
col2.metric("Orders", f"{orders:,}")
col3.metric("Units", f"{units:,}" if not np.isnan(units) else "—")
col4.metric("Avg Order Value", f"${aov:,.0f}" if not np.isnan(aov) else "—")
col5.metric("Margin %", f"{margin*100:,.1f}%" if not np.isnan(margin) else "—")

st.divider()

# -----------------------------
# ---------- CHARTS -----------
# -----------------------------

if DF.empty:
    st.warning("No data after filters. Try expanding your date range or selections.")
    st.stop()

# Revenue over time
gran = auto_granularity(pd.to_datetime(start_date), pd.to_datetime(end_date))
series = DF.set_index("order_date").sort_index().resample(gran)["sales"].sum().reset_index()
fig_ts = px.line(series, x="order_date", y="sales", title=f"Revenue over time ({'Daily' if gran=='D' else 'Weekly' if gran=='W' else 'Monthly'})")
fig_ts.update_yaxes(tickprefix="$", separatethousands=True)
st.plotly_chart(fig_ts, use_container_width=True)

# Row of two charts: Category & Channel
c1, c2 = st.columns(2)
if "category" in DF.columns:
    cat = DF.groupby("category", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    fig_cat = px.bar(cat, x="category", y="sales", title="Sales by Category")
    fig_cat.update_yaxes(tickprefix="$", separatethousands=True)
    c1.plotly_chart(fig_cat, use_container_width=True)

if "channel" in DF.columns:
    ch = DF.groupby("channel", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    fig_ch = px.pie(ch, names="channel", values="sales", title="Channel Mix")
    c2.plotly_chart(fig_ch, use_container_width=True)

# Top products (Pareto)
if "product" in DF.columns:
    prod = DF.groupby("product", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    prod["cum_share"] = prod["sales"].cumsum() / prod["sales"].sum()
    top_n = min(20, len(prod))
    fig_prod = px.bar(prod.head(top_n), x="product", y="sales", title=f"Top {top_n} Products (Pareto)")
    fig_prod.update_yaxes(tickprefix="$", separatethousands=True)
    fig_prod.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig_prod, use_container_width=True)

# Region/Country table
geo_cols = [c for c in ["region", "country"] if c in DF.columns]
if geo_cols:
    st.subheader("Geography performance")
    geo = DF.groupby(geo_cols, as_index=False).agg(
        Revenue=("sales", "sum"),
        Orders=("_orders", "nunique"),
        Units=("quantity", "sum") if "quantity" in DF.columns else ("sales", "count"),
        Profit=("profit", "sum") if DF["profit"].notna().any() else ("sales", "sum")
    )
    geo["AOV"] = geo["Revenue"] / geo["Orders"].replace(0, np.nan)
    if DF["profit"].notna().any():
        geo["Margin %"] = np.where(geo["Revenue"]>0, geo["Profit"]/geo["Revenue"]*100, np.nan)
    st.dataframe(geo.sort_values("Revenue", ascending=False), use_container_width=True)

# Detail table
with st.expander("🔎 Show filtered dataset"):
    st.dataframe(DF.sort_values("order_date"), use_container_width=True)

# Download
csv = DF.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download filtered CSV", data=csv, file_name="filtered_sales.csv", mime="text/csv")

# Help panel
with st.expander("❓ Data prep & tips"):
    st.markdown(
        """
        **Minimum**: An order date and either a `Sales` column or both `Quantity` and `Unit Price`.
        
        **Optional** (but recommended): Order ID, Product, Category, Channel, Region/Country, Discount, Cost.
        
        If your columns don't match exactly, common variants are automatically recognized (e.g., `OrderDate` or `invoice_date`).
        """
    )

# -----------------------------
# ------ SAMPLE EXPORT --------
# -----------------------------
with st.sidebar.expander("Need a template?"):
    demo = load_demo_data(1000)
    st.caption("Download a template CSV mirroring the demo schema.")
    st.download_button(
        "Download template.csv",
        data=demo.head(200).to_csv(index=False).encode("utf-8"),
        file_name="template_sales.csv",
        mime="text/csv",
    )
