import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import quantstats as qs
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

import yfinance as yf

def fetch_stock_data(tickers, start_date='2010-07-01', end_date='2023-02-10'):
    """Fetch historical stock data from Yahoo Finance."""
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        
        # Print available columns to check if 'Adj Close' exists
        print("Available columns:", data.columns)
        
        # Use 'Adj Close' if available, otherwise fallback to 'Close'
        if 'Adj Close' in data.columns:
            data = data['Adj Close']
        else:
            print("Warning: 'Adj Close' not found, using 'Close' instead.")
            data = data['Close']
        
        returns = data.pct_change().dropna()
        return returns
    except Exception as e:
        return f"Error fetching stock data: {str(e)}"

def portfolio_risk(weights, cov_matrix):
    """Calculate portfolio risk (standard deviation)."""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def optimize_weights(returns):
    """Optimize portfolio weights using mean-variance optimization."""
    try:
        cov_matrix = returns.cov()
        num_assets = len(returns.columns)
        initial_weights = np.ones(num_assets) / num_assets
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # Sum of weights = 1
        bounds = tuple((0, 1) for _ in range(num_assets))  # Weights between 0 and 1
        optimized = minimize(portfolio_risk, initial_weights, args=(cov_matrix,),
                             method='SLSQP', bounds=bounds, constraints=constraints)
        return optimized.x if optimized.success else "Optimization failed"
    except Exception as e:
        return str(e)

st.set_page_config(page_title="Portfolio Optimizer", page_icon="📈", layout="wide")

st.sidebar.title("📊 Portfolio Optimizer")
st.sidebar.write("Enter stock tickers separated by commas (e.g., AAPL, TSLA, GOOGL).")
user_input = st.sidebar.text_input("Stock Tickers:", "AAPL, TSLA, DIS, AMD, GOOGL")

if st.sidebar.button("Optimize Portfolio"):
    tickers = [t.strip().upper() for t in user_input.split(',')]
    returns = fetch_stock_data(tickers)

    if isinstance(returns, str):
        st.sidebar.error(f"Error fetching stock data: {returns}")
    else:
        optimized_weights = optimize_weights(returns)

        if isinstance(optimized_weights, str):
            st.sidebar.error(optimized_weights)
        else:
            st.subheader("📊 Optimized Portfolio Weights")
            col1, col2 = st.columns(2)
            with col1:
                for stock, weight in zip(returns.columns, optimized_weights):
                    st.write(f"**{stock}:** {weight:.2%}")
            
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(optimized_weights, labels=returns.columns, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
            ax.set_title("Optimized Portfolio Allocation")
            st.pyplot(fig)
            
            report_path = "analysis_report.html"
            qs.reports.html(returns, output=report_path, title="Portfolio Analysis", benchmark="SPY")

            st.subheader("📈 Stock Return Data")
            st.dataframe(returns.tail(10))

            st.subheader("📉 Historical Performance")
            fig, ax = plt.subplots(figsize=(10, 4))
            returns.cumsum().plot(ax=ax)
            ax.set_title("Cumulative Returns Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Cumulative Returns")
            st.pyplot(fig)

            with open(report_path, "r", encoding="utf-8") as f:
                report_html = f.read()

            st.subheader("📄 Portfolio Analysis Report")
            st.components.v1.html(report_html, height=600, scrolling=True)

            with open(report_path, "rb") as f:
                report_bytes = f.read()

            st.download_button(label="📥 Download Report",
                               data=report_bytes,
                               file_name="Portfolio_Analysis_Report.html",
                               mime="text/html")
