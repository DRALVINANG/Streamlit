import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import pyfolio as pf
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import streamlit as st
from datetime import datetime
from tabulate import tabulate


# Function to get target features
def get_target_features(data):
    data['PCT_CHANGE'] = data['Close'].pct_change()
    data['VOLATILITY'] = data['PCT_CHANGE'].rolling(14).std() * 100
    data['SMA'] = ta.sma(data['Close'], timeperiod=14)
    data['CORR'] = data['Close'].rolling(14).corr(data['SMA'])
    data['RSI'] = ta.rsi(data['Close'], timeperiod=14)
    data['ADX'] = ta.adx(data['High'], data['Low'], data['Close'], length=14)['ADX_14']
    data['Returns_4_Tmrw'] = data['Close'].pct_change().shift(-1)
    data['Actual_Signal'] = np.where(data['Returns_4_Tmrw'] > 0, 1, 0)
    data = data.dropna()
    return data['Actual_Signal'], data[['VOLATILITY', 'CORR', 'RSI', 'ADX']]

# Left Sidebar for ticker input
ticker = st.sidebar.text_input("Enter the Ticker Symbol", 'D05.SI')

# User input for training and backtesting period
train_start_date = st.sidebar.date_input("Training Start Date", datetime(2022, 1, 1))
train_end_date = st.sidebar.date_input("Training End Date", datetime(2022, 12, 31))
backtest_start_date = st.sidebar.date_input("Backtesting Start Date", datetime(2023, 1, 1))
backtest_end_date = st.sidebar.date_input("Backtesting End Date", datetime(2024, 1, 1))

# Dynamically update title with company's name (based on ticker)
company_name = yf.Ticker(ticker).info.get('longName', ticker)  # Get company name from Yahoo Finance
st.title(f'🌳Stock Prediction and Backtesting for {company_name} using Random Forest Classifier (RFC)')

# Download stock data for training
data = yf.download(ticker, start=train_start_date, end=train_end_date)
data.columns = data.columns.droplevel(level=1)

# Plot Close Price
st.subheader('Stock Price Close Plot')
fig, ax = plt.subplots(figsize=(18, 5))
data['Close'].plot(ax=ax, color='b')
ax.set_ylabel('Close Price')
ax.set_xlabel('Date')
ax.set_title(f'{ticker} Close Price')
st.pyplot(fig)

# Get target and features
y, X = get_target_features(data)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Scale features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train Random Forest Classifier model
rfc = RandomForestClassifier(n_estimators=10, criterion="entropy", max_depth=3, max_features=3)
model = rfc.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Accuracy score as percentage
accuracy = metrics.accuracy_score(y_test, y_pred) * 100
st.subheader('Model Accuracy')
st.write(f"Accuracy: {accuracy:.2f}%")

# Description
st.write("The accuracy score represents how well the model predicts the direction of stock price movement based on the features used.")
st.markdown("---")

# Backtesting the strategy
df_backtest = yf.download(ticker, start=backtest_start_date, end=backtest_end_date)
df_backtest.columns = df_backtest.columns.droplevel(level=1)

# Feature creation for backtesting
df_backtest['PCT_CHANGE'] = df_backtest['Close'].pct_change()
df_backtest['VOLATILITY'] = df_backtest['PCT_CHANGE'].rolling(14).std() * 100
df_backtest['RSI'] = ta.rsi(df_backtest['Close'], timeperiod=14)
df_backtest['ADX'] = ta.adx(df_backtest['High'], df_backtest['Low'], df_backtest['Close'], length=14)['ADX_14']
df_backtest['SMA'] = ta.sma(df_backtest['Close'], timeperiod=14)
df_backtest['CORR'] = df_backtest['Close'].rolling(window=14).corr(df_backtest['SMA'])
df_backtest = df_backtest.dropna()

# Scale features for backtesting
df_scaled_backtest = sc.transform(df_backtest[['VOLATILITY', 'CORR', 'RSI', 'ADX']])
df_backtest['predicted_signal_4_tmrw'] = model.predict(df_scaled_backtest)

# Calculate strategy returns
df_backtest['strategy_returns'] = df_backtest['predicted_signal_4_tmrw'] * df_backtest['PCT_CHANGE'].shift(-1)
df_backtest.dropna(inplace=True)

# Display Pyfolio performance stats
st.subheader('Performance Stats')
perf_stats = pf.timeseries.perf_stats(df_backtest.strategy_returns)

# Display performance stats line by line
for stat_name, stat_value in perf_stats.items():
    st.write(f"**{stat_name}:** {stat_value:.4f}")

st.markdown("---")
st.write("The performance stats provide insights into how the strategy performed over time, such as cumulative return, volatility, and maximum drawdown.")
st.markdown("---")

# Display Pyfolio tear sheet
st.subheader(f'{ticker} Pyfolio Tear Sheet')
if not df_backtest['strategy_returns'].empty:
    try:
        pf.create_simple_tear_sheet(df_backtest['strategy_returns'])
        st.pyplot(plt)  # Displaying the tear sheet using matplotlib
    except Exception as e:
        st.error(f"Error generating tear sheet: {e}")
else:
    st.error("Invalid stock returns data. Please check the ticker and try again.")

# Add conclusion section
st.markdown("---")
st.write("### CONCLUSION")

# Extract the necessary stats for the conclusion
cumulative_returns = perf_stats.get('Cumulative returns', 'N/A')

# Ensure the cumulative return is numeric, otherwise use 'N/A'
try:
    cumulative_returns = float(cumulative_returns) * 100  # Convert to percentage
    cumulative_returns = f"{cumulative_returns:.2f}%"
except (ValueError, TypeError):
    cumulative_returns = 'N/A'

# Conclusion with bold and larger font for all points
st.markdown(f"""
<h2 style="font-size: 24px; font-weight: bold;">Machine Learning Model used:
<h2 style="font-size: 18px; font-weight: bold; color: red;">-  Random Forest Classifier (RFC)</h2>
 
<h2 style="font-size: 24px; font-weight: bold;">Training Period:
<h2 style="font-size: 18px; font-weight: bold; color: red;">- {train_start_date} to {train_end_date}</h2>

<h2 style="font-size: 24px; font-weight: bold;">Backtesting Period:
<h2 style="font-size: 18px; font-weight: bold; color: red;"> - {backtest_start_date} to {backtest_end_date}</h2>

<h2 style="font-size: 24px; font-weight: bold;">Features / Technical Indicators used:</h2>
<h3 style="font-size: 18px; font-weight: bold; color: red;">  - ADX (Average Directional Index)</h3>
<h3 style="font-size: 18px; font-weight: bold; color: red;">  - Volatility</h3>
<h3 style="font-size: 18px; font-weight: bold; color: red;">  - Correlation between price and SMA</h3>
<h3 style="font-size: 18px; font-weight: bold; color: red;">  - RSI (Relative Strength Index)</h3>
<h2 style="font-size: 24px; font-weight: bold;">Model accuracy:
<h2 style="font-size: 48px; font-weight: bold; color: red;">  {accuracy:.2f}%</h2>

<h2 style="font-size: 24px; font-weight: bold;">Cumulative Returns during the Backtesting Period:
<h2 style="font-size: 88px; font-weight: bold; color: red;">  {cumulative_returns}</h2>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.write("Created by Dr. Alvin Ang")

