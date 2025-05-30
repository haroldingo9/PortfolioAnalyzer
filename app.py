import streamlit as st
import statistics
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import base64

st.title("📊 Portfolio Performance Dashboard")


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
price_template = """AAPL,MSFT,GOOGL,TSLA,AMZN
120,220,1500,600,3100
125,225,1520,580,3200
130,230,1550,590,3300
140,250,1650,650,3350
145,255,1750,680,3380
150,260,1800,700,3400
"""

st.markdown(file_download_link("portfolio_template.csv", portfolio_template), unsafe_allow_html=True)
st.markdown(file_download_link("monthly_prices_template.csv", price_template), unsafe_allow_html=True)


uploaded_portfolio = st.file_uploader("Upload Portfolio CSV", type=["csv"])
uploaded_prices = st.file_uploader("Upload Monthly Prices CSV (optional)", type=["csv"])

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

if uploaded_prices is not None:
    df_prices = pd.read_csv(uploaded_prices)
    try:
        monthly_prices = {
            stock: df_prices[stock].dropna().astype(float).tolist()
            for stock in df_prices.columns
        }
        st.success("✅ Monthly prices uploaded successfully!")
    except Exception as e:
        st.error(f"Error processing monthly prices CSV: {e}")
        monthly_prices = None
else:
    monthly_prices = None
    st.info("Upload monthly prices CSV to enable risk analysis.")


risk_free_rate_annual = 6
risk_free_rate_monthly = risk_free_rate_annual / 12


total_invested = 0
total_current_value = 0
returns = {}
profits = {}

st.subheader("💸 Individual Stock Investment and Returns")
for stock, info in portfolio.items():
    qty = info["quantity"]
    buy = info["buy_price"]
    current = info["current_price"]
    invested = qty * buy
    current_value = qty * current
    profit = current_value - invested
    ret = (profit / invested) * 100 if invested != 0 else 0

    total_invested += invested
    total_current_value += current_value
    returns[stock] = ret
    profits[stock] = profit

    st.markdown(f"**{stock}**: Invested ₹{invested}, Current ₹{current_value}, Profit ₹{profit:.2f}, Return {ret:.2f}%")

total_return = ((total_current_value - total_invested) / total_invested) * 100 if total_invested != 0 else 0
st.markdown(f"**Total Invested:** ₹{total_invested}")
st.markdown(f"**Total Current Value:** ₹{total_current_value}")
st.markdown(f"**Portfolio Return:** {total_return:.2f}%")


st.subheader("📌 Profit Contribution to Portfolio")
total_profit = sum(profits.values())
for stock, profit in profits.items():
    contribution = (profit / total_profit) * 100 if total_profit != 0 else 0
    st.markdown(f"{stock}: {contribution:.2f}% of total profit")


def max_drawdown(prices):
    peak = prices[0]
    max_dd = 0
    for price in prices:
        if price > peak:
            peak = price
        dd = (peak - price) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return max_dd

def var_cvar(returns, confidence_level=0.95):
    sorted_returns = sorted(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    VaR = abs(sorted_returns[index])
    CVaR = abs(np.mean(sorted_returns[:index + 1]))
    return VaR, CVaR

sharpe_ratios = {}

if monthly_prices is not None:
    st.subheader("⚠️ Risk Analysis per Stock")
    for stock, prices in monthly_prices.items():
        if len(prices) < 2:
            st.warning(f"Not enough data points for {stock} to calculate risk metrics.")
            continue

        monthly_returns = [(prices[i] - prices[i - 1]) / prices[i - 1] * 100 for i in range(1, len(prices))]

        volatility = statistics.stdev(monthly_returns)
        avg_return = statistics.mean(monthly_returns)
        sharpe = (avg_return - risk_free_rate_monthly) / volatility if volatility != 0 else 0
        sharpe_ratios[stock] = sharpe
        mdd = max_drawdown(prices)
        VaR, CVaR = var_cvar(monthly_returns)

        st.markdown(f"**{stock}** → Volatility: {volatility:.2f}%, Avg Return: {avg_return:.2f}%, Sharpe Ratio: {sharpe:.2f}")
        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Max Drawdown: {mdd:.2f}%, VaR(95%): {VaR:.2f}%, CVaR(95%): {CVaR:.2f}%")
else:
    st.info("Monthly prices data not provided. Skipping risk analysis.")


if monthly_prices is not None:
    returns_df = pd.DataFrame({
        stock: [(monthly_prices[stock][i] - monthly_prices[stock][i - 1]) / monthly_prices[stock][i - 1] * 100 for i in range(1, len(monthly_prices[stock]))]
        for stock in portfolio if stock in monthly_prices
    })

    st.subheader("📈 Correlation Matrix")
    st.dataframe(returns_df.corr())
else:
    st.info("Monthly prices data not provided. Skipping correlation matrix.")


if monthly_prices is not None and len(portfolio) > 0:
    expected_returns = returns_df.mean().values / 100
    cov_matrix = returns_df.cov().values / 10000
    num_assets = len(portfolio)
    weights_init = np.array([profits[stock] / total_current_value if total_current_value != 0 else 1/num_assets for stock in portfolio])

    def portfolio_variance(weights, cov_matrix):
        return weights.T @ cov_matrix @ weights

    def portfolio_return(weights, expected_returns):
        return weights.T @ expected_returns

    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: portfolio_return(w, expected_returns) - (total_return / 100) / 6}
    )

    bounds = tuple((0, 1) for _ in range(num_assets))

    opt = minimize(portfolio_variance, weights_init, args=(cov_matrix,), method='SLSQP',
                   bounds=bounds, constraints=constraints)

    st.subheader("🛠️ Portfolio Optimization Suggestion")
    if opt.success:
        optimized_weights = opt.x
        for i, stock in enumerate(portfolio):
            st.markdown(f"{stock}: Current weight = {weights_init[i]*100:.2f}%, Optimized weight = {optimized_weights[i]*100:.2f}%")
    else:
        st.markdown("**❌ Optimization Failed.**")
else:
    st.info("Monthly prices data required for portfolio optimization.")


if sharpe_ratios:
    best_sharpe = max(sharpe_ratios, key=sharpe_ratios.get)
    worst_sharpe = min(sharpe_ratios, key=sharpe_ratios.get)

    st.subheader("🏆 Sharpe Ratio Ranking")
    st.markdown(f"**Best Stock:** {best_sharpe} ({sharpe_ratios[best_sharpe]:.2f})")
    st.markdown(f"**Worst Stock:** {worst_sharpe} ({sharpe_ratios[worst_sharpe]:.2f})")
else:
    st.info("Sharpe ratios not available (monthly prices missing).")