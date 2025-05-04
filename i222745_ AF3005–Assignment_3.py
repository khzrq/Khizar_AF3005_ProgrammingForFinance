

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

st.sidebar.image("data:image/jpeg;base64,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", use_column_width=True)
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
