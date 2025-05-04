import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
import quantstats as qs
import os

# -------------------------------
# Page Config & Session State Init
st.set_page_config(page_title="Finance ML App", layout="wide")

if "step" not in st.session_state:
    st.session_state.step = 0
if "df" not in st.session_state:
    st.session_state.df = None
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

# -------------------------------
# 🧭 Sidebar: Upload or Fetch
st.sidebar.image("https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif", use_column_width=True)
st.sidebar.markdown("## Upload Kragle Dataset")
kragle_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

st.sidebar.markdown("## Or Fetch from Yahoo Finance")
ticker_input = st.sidebar.text_input("Enter a Stock Ticker (e.g., AAPL, TSLA)", "AAPL")
fetch_data = st.sidebar.button("Fetch Data")

st.title("💹 Financial ML Dashboard")

# -------------------------------
# 📂 Step 1: Load Data
if kragle_file is not None:
    st.session_state.df = pd.read_csv(kragle_file)
    st.success("✅ Kragle dataset loaded successfully!")
    st.dataframe(st.session_state.df.head())
    st.session_state.step = 1

elif fetch_data:
    try:
        data = yf.download(ticker_input, start="2010-01-01", end="2024-12-31")
        st.session_state.df = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        st.success(f"✅ {ticker_input} data fetched successfully!")
        st.dataframe(st.session_state.df.head())
        st.session_state.step = 1
    except:
        st.error("❌ Failed to fetch Yahoo Finance data.")

# -------------------------------
# 🧹 Step 2: Preprocessing
if st.session_state.step >= 1:
    if st.button("Run Preprocessing"):
        st.session_state.df.dropna(inplace=True)
        st.success("✅ Missing values removed!")
        st.dataframe(st.session_state.df.describe())
        st.session_state.step = 2

# -------------------------------
# 🧠 Step 3: Train Model
if st.session_state.step >= 2:
    if st.button("Train ML Model"):
        df = st.session_state.df.copy()
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)

        X = df[['Open', 'High', 'Low', 'Volume']]
        y = df['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        st.session_state.model_trained = True
        st.session_state.model_score = model.score(X_test, y_test)
        st.session_state.y_test = y_test
        st.session_state.predictions = predictions

        st.success("✅ Model trained successfully!")
        st.write(f"**R² Score:** {st.session_state.model_score:.2f}")

        fig = px.scatter(x=y_test, y=predictions, labels={'x': 'Actual', 'y': 'Predicted'}, title="📈 Actual vs Predicted")
        st.plotly_chart(fig)

# -------------------------------
# 📄 Step 4: Generate PDF Report
if st.session_state.model_trained:
    if st.button("📄 Generate PDF Report"):
        qs.reports.html(returns, output='analysis_report.html', title='Financial Report', benchmark='SPY')
        st.success("✅ QuantStats HTML report generated!")

        with open("analysis_report.html", "rb") as file:
            st.download_button("📥 Download Report", file, "analysis_report.html", mime="text/html")
