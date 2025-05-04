

# 🚀 How to Run
# 1. Install dependencies:
# pip install -r requirements.txt
# 2. Run with:
# streamlit run app.py


import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
import matplotlib.pyplot as plt
import quantstats as qs
import os



st.set_page_config(page_title="Finance ML App", layout="wide")
st.title("💹 Financial ML Dashboard")

st.sidebar.image("https://www.istockphoto.com/fi/valokuva/currency-and-exchange-stock-chart-for-finance-and-economy-display-gm1956949830-557559253", use_column_width=True)
st.sidebar.markdown("## Upload Kragle Dataset")
kragle_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

st.sidebar.markdown("## Or Fetch from Yahoo Finance")
ticker_input = st.sidebar.text_input("Enter a Stock Ticker (e.g., AAPL, TSLA)", "AAPL")
fetch_data = st.sidebar.button("Fetch Data")

# -------------------------------
# 📂 Step 1: Load Data
df = None
if kragle_file is not None:
    df = pd.read_csv(kragle_file)
    st.success("✅ Kragle dataset loaded successfully!")
    st.dataframe(df.head())

elif fetch_data:
    try:
        data = yf.download(ticker_input, start="2010-01-01", end="2024-12-31")
        df = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        st.success(f"✅ {ticker_input} data fetched successfully!")
        st.dataframe(df.head())
    except:
        st.error("❌ Failed to fetch Yahoo Finance data.")

# -------------------------------
# 🧹 Step 2: Preprocessing
if df is not None:
    st.markdown("## 🧹 Data Preprocessing")
    if st.button("Run Preprocessing"):
        df.dropna(inplace=True)
        st.success("✅ Missing values removed!")
        st.write("Remaining data stats:")
        st.dataframe(df.describe())

# -------------------------------
# 🛠️ Step 3: Train Model
if df is not None and st.button("Train ML Model"):
    st.markdown("## 🧠 ML Model: Linear Regression")

    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    st.success("✅ Model trained successfully!")
    st.write(f"**R² Score:** {model.score(X_test, y_test):.2f}")

    fig = px.scatter(x=y_test, y=predictions, labels={'x': 'Actual', 'y': 'Predicted'}, title="📈 Actual vs Predicted")
    st.plotly_chart(fig)

# -------------------------------
# 📊 Step 4: Generate PDF Report
if df is not None and st.button("📄 Generate PDF Report"):
    returns = df['Close'].pct_change().dropna()
    qs.reports.html(returns, output='analysis_report.html', title='Financial Report', benchmark='SPY')
    st.success("✅ QuantStats HTML report generated!")

    with open("analysis_report.html", "rb") as file:
        st.download_button("📥 Download Report", file, "analysis_report.html", mime="text/html")
