import streamlit as st
import pandas as pd
import numpy as np
import statistics
from scipy.optimize import minimize
import base64
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Portfolio Analyzer", layout="wide")
st.title("📊 Portfolio Performance Dashboard")

# ------------------ 🔻 Disclaimer Section 🔻 ------------------ #
st.markdown("""
> ℹ️ **Disclaimer**: This dashboard is for educational and informational purposes only.  
> It does not constitute financial advice. Always consult a certified financial advisor before making investment decisions.  
> The data provided is based on user inputs and publicly available sources like Yahoo Finance.
""")

# ------------------ 🔻 File Download Links 🔻 ------------------ #
def file_download_link(file_name, content):
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">📄 Download Template: {file_name}</a>'
    return href

portfolio_template = """Stock,Quantity,Buy_Price,Current_Price
AAPL,50,120,150
MSFT,30,220,260
GOOGL,10,1500,1800
TSLA,20,600,700
AMZN,5,3100,3400
"""

st.markdown(file_download_link("portfolio_template.csv", portfolio_template), unsafe_allow_html=True)

# ------------------ 🔻 Upload Files 🔻 ------------------ #
uploaded_portfolio = st.file_uploader("📥 Upload Portfolio CSV", type=["csv"])

# ------------------ 🔻 Portfolio Data Process 🔻 ------------------ #
if uploaded_portfolio is not None:
    df_portfolio = pd.read_csv(uploaded_portfolio)
    try:
        portfolio = {
            row["Stock"]: {
                "quantity": float(row["Quantity"]),
                "buy_price": float(row["Buy_Price"]),
                "current_price": float(row["Current_Price"])
            }
            for _, row in df_portfolio.iterrows()
        }
        st.success("✅ Portfolio data uploaded successfully!")
    except Exception as e:
        st.error(f"Error processing portfolio CSV: {e}")
        st.stop()
else:
    st.info("Please upload your portfolio CSV file to proceed.")
    st.stop()

# ------------------ 🔻 Auto-fetch Monthly Prices from Yahoo Finance 🔻 ------------------ #
monthly_prices = {}
end = datetime.today()
start = end - timedelta(days=30*6)

with st.spinner("📡 Fetching 6-month closing prices from Yahoo Finance..."):
    for stock in portfolio.keys():
        try:
            data = yf.download(stock, start=start, end=end, interval="1mo", progress=False)
            if not data.empty:
                closes = data['Close']
                if isinstance(closes, pd.DataFrame):  # In case of multi-indexed columns
                    closes = closes[stock]
                closes = closes.dropna().values.tolist()
                if len(closes) >= 2:
                    monthly_prices[stock] = closes
        except Exception as e:
            st.error(f"Error fetching data for {stock}: {e}")

if not monthly_prices:
    st.warning("⚠️ Could not retrieve any monthly price data. Skipping risk analysis and optimization.")

# ------------------ 🔻 Portfolio Performance 🔻 ------------------ #
st.subheader("💸 Individual Stock Investment and Returns")
total_invested = 0
total_current_value = 0
returns = {}
profits = {}

for stock, info in portfolio.items():
    qty = info["quantity"]
    buy = info["buy_price"]
    current = info["current_price"]
    invested = qty * buy
    current_value = qty * current
    profit = current_value - invested
    ret = (profit / invested) * 100 if invested else 0

    total_invested += invested
    total_current_value += current_value
    returns[stock] = ret
    profits[stock] = profit

    st.markdown(f"**{stock}**: Invested ₹{invested:.2f}, Current ₹{current_value:.2f}, Profit ₹{profit:.2f}, Return {ret:.2f}%")

total_return = ((total_current_value - total_invested) / total_invested) * 100 if total_invested else 0
st.markdown(f"**Total Invested:** ₹{total_invested:.2f}")
st.markdown(f"**Total Current Value:** ₹{total_current_value:.2f}")
st.markdown(f"**Portfolio Return:** {total_return:.2f}%")

# ------------------ 🔻 Profit Contribution 🔻 ------------------ #
st.subheader("📌 Profit Contribution to Portfolio")
total_profit = sum(profits.values())
for stock, profit in profits.items():
    contribution = (profit / total_profit) * 100 if total_profit else 0
    st.markdown(f"{stock}: {contribution:.2f}% of total profit")

# ------------------ 🔻 Risk Metrics Functions 🔻 ------------------ #
def max_drawdown(prices):
    peak = prices[0]
    max_dd = 0
    for price in prices:
        if price > peak:
            peak = price
        dd = (peak - price) /



