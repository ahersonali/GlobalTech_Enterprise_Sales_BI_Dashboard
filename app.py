# ================================
# ENTERPRISE SALES BI DASHBOARD
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import random
from faker import Faker
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(layout="wide")
st.title("🏢 GlobalTech Retail Corp - Enterprise BI Dashboard")

# -----------------------------
# DATA GENERATION (5000+ rows)
# -----------------------------
@st.cache_data
def generate_data(n=5000):
    fake = Faker()
    np.random.seed(42)

    regions = ["North", "South", "East", "West"]
    categories = ["Technology", "Furniture", "Office Supplies"]
    segments = ["Corporate", "SMB", "Enterprise"]
    shipping_modes = ["Standard", "Express", "Same Day"]
    sales_channels = ["Online", "Offline"]

    products = {
        "Technology": ["Laptop", "Phone", "Tablet", "Monitor"],
        "Furniture": ["Chair", "Desk", "Cabinet"],
        "Office Supplies": ["Notebook", "Printer Paper", "Stapler"]
    }

    data = []

    for i in range(n):
        category = random.choice(categories)
        product = random.choice(products[category])
        sales = np.random.randint(100, 5000)
        quantity = np.random.randint(1, 10)
        discount = round(np.random.uniform(0, 0.3), 2)
        profit = round(sales * np.random.uniform(0.05, 0.25), 2)

        data.append([
            10000 + i,
            fake.date_between(start_date='-2y', end_date='today'),
            random.choice(regions),
            category,
            product,
            random.choice(segments),
            random.choice(shipping_modes),
            random.choice(sales_channels),
            sales,
            quantity,
            discount,
            profit
        ])

    df = pd.DataFrame(data, columns=[
        "Order_ID", "Order_Date", "Region", "Category",
        "Product", "Segment", "Shipping_Mode",
        "Sales_Channel", "Sales", "Quantity",
        "Discount", "Profit"
    ])

    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    return df


df = generate_data()

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("🔎 Filters")

region_filter = st.sidebar.multiselect(
    "Select Region",
    df["Region"].unique(),
    default=df["Region"].unique()
)

segment_filter = st.sidebar.multiselect(
    "Select Segment",
    df["Segment"].unique(),
    default=df["Segment"].unique()
)

df = df[
    (df["Region"].isin(region_filter)) &
    (df["Segment"].isin(segment_filter))
]

# -----------------------------
# KPI METRICS
# -----------------------------
total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
profit_margin = (total_profit / total_sales) * 100
avg_discount = df["Discount"].mean() * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Sales", f"${total_sales:,.0f}")
col2.metric("Total Profit", f"${total_profit:,.0f}")
col3.metric("Profit Margin", f"{profit_margin:.2f}%")
col4.metric("Avg Discount", f"{avg_discount:.2f}%")

# -----------------------------
# MONTHLY SALES TREND
# -----------------------------
monthly_sales = df.resample('M', on='Order_Date')['Sales'].sum().reset_index()

fig1 = px.line(
    monthly_sales,
    x="Order_Date",
    y="Sales",
    title="📈 Monthly Sales Trend"
)

st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# PIVOT TABLE HEATMAP
# -----------------------------
pivot = pd.pivot_table(
    df,
    values="Sales",
    index="Region",
    columns="Category",
    aggfunc="sum"
)

fig2 = px.imshow(
    pivot,
    text_auto=True,
    title="🔥 Sales Heatmap (Region vs Category)"
)

st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# TOP PRODUCTS
# -----------------------------
top_products = (
    df.groupby("Product")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig3 = px.bar(
    top_products,
    x="Sales",
    y="Product",
    orientation="h",
    title="🏆 Top 10 Products by Sales"
)

st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# ARIMA FORECASTING
# -----------------------------
st.subheader("🔮 6-Month Sales Forecast (ARIMA)")

monthly_series = df.resample('M', on='Order_Date')['Sales'].sum()

if len(monthly_series) > 6:
    model = ARIMA(monthly_series, order=(1,1,1))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=6)

    forecast_df = pd.DataFrame({
        "Date": pd.date_range(
            start=monthly_series.index[-1],
            periods=7,
            freq='M'
        )[1:],
        "Forecast": forecast
    })

    fig4 = px.line(
        forecast_df,
        x="Date",
        y="Forecast",
        title="📊 Forecasted Revenue"
    )

    st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# DATA PREVIEW
# -----------------------------
st.subheader("📄 Raw Enterprise Data Preview")
st.dataframe(df.head(20))
