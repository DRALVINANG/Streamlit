import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import pyfolio as pf
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression  # Multiple Linear Regression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import streamlit as st
from tabulate import tabulate
from datetime import datetime

# Set Pandas options for display
pd.set_option('display.max_columns', 20)

# Function to get target features
def get_target_features(data):
    # Define Features (X)
    data['PCT_CHANGE'] = data['Close'].pct_change()
    data['VOLATILITY'] = data['PCT_CHANGE'].rolling(14).std() * 100
    data['RSI'] = ta.rsi(data['Close'], timeperiod=14)
    data['ADX'] = ta.adx(data['High'], data['Low'], data['Close'], length=14)['ADX_14']
    data['SMA'] = ta.sma(data['Close'], timeperiod=14)
    data['CORR'] = data['Close'].rolling(window=14).corr(data['SMA'])

    # Define Target (y)
    data['Returns_4_Tmrw'] = data['Close'].pct_change().shift(-1)
    data['Actual_Signal'] = np.where(data['Returns_4_Tmrw'] > 0, 1, 0)

    # Drop NaN rows
    data = data.dropna()
    data = data.round(5)

    return data['Actual_Signal'], data[['VOLATILITY', 'CORR', 'RSI', 'ADX']]

# Streamlit Sidebar for User Input
ticker = st.sidebar.text_input("Enter the Ticker Symbol", 'D05.SI')

train_start_date = st.sidebar.date_input("Training Start Date", datetime(2020, 1, 1))
train_end_date = st.sidebar.date_input("Training End Date", datetime(2021, 1, 1))

backtest_start_date = st.sidebar.date_input("Backtesting Start Date", datetime(2021, 1, 1))
backtest_end_date = st.sidebar.date_input("Backtesting End Date", datetime(2022, 1, 1))

# Dynamically Update Title
company_name = yf.Ticker(ticker).info.get('longName', ticker)
st.title(f'📊Multiple Linear Regression for {company_name}')

# Download Stock Data
data = yf.download(ticker, start=train_start_date, end=train_end_date)
data.columns = data.columns.droplevel(level=1)

# Plotting the Close Price
st.subheader('Stock Price Close Plot')
fig, ax = plt.subplots(figsize=(18, 5))
data['Close'].plot(ax=ax, color='b')
ax.set_ylabel('Close Price')
ax.set_xlabel('Date')
ax.set_title(f'{ticker} Close Price')
st.pyplot(fig)

st.write("This plot shows the closing price of the selected stock over the specified training period.")
st.markdown('---')

# Get target and features
y, X = get_target_features(data)

# Split data into train and test sets
split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Scale the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the Multiple Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Convert continuous predictions to binary signals (1 or 0)
y_pred = np.where(y_pred > 0.5, 1, 0)

# Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred) * 100
st.subheader('Model Accuracy')
st.write(f"Accuracy: {accuracy:.2f}%")

st.write("This section evaluates the model's performance by displaying the accuracy, which represents the percentage of correct predictions.")
st.markdown('---')

# Backtesting the Model
df = yf.download(ticker, start=backtest_start_date, end=backtest_end_date)
df.columns = df.columns.droplevel(level=1)

# Feature Engineering for Backtesting
df['PCT_CHANGE'] = df['Close'].pct_change()
df['VOLATILITY'] = df['PCT_CHANGE'].rolling(14).std() * 100
df['RSI'] = ta.rsi(df['Close'], timeperiod=14)
df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
df['SMA'] = ta.sma(df['Close'], timeperiod=14)
df['CORR'] = df['Close'].rolling(window=14).corr(df['SMA'])
df = df.dropna()

# Scale the backtest data and predict
df_scaled = sc.transform(df[['VOLATILITY', 'CORR', 'RSI', 'ADX']])
y_pred_continuous = model.predict(df_scaled)

# Convert continuous predictions into binary signals (1 or 0)
df['predicted_signal_4_tmrw'] = np.where(y_pred_continuous > 0.5, 1, 0)

# Calculate Strategy Returns
df['strategy_returns'] = df['predicted_signal_4_tmrw'].shift(1) * df['PCT_CHANGE']
df.dropna(inplace=True)

# Performance Stats using Pyfolio
st.subheader('Performance Stats')
perf_stats = pf.timeseries.perf_stats(df.strategy_returns)
for stat_name, stat_value in perf_stats.items():
    st.write(f"**{stat_name}:** {stat_value:.4f}")


st.write("This section shows the performance statistics of the backtested strategy, such as annualized returns, Sharpe ratio, and drawdown.")
st.markdown('---')

# Display Pyfolio Tear Sheet
st.subheader(f'{ticker} Pyfolio Tear Sheet')
try:
    pf.create_simple_tear_sheet(df['strategy_returns'])
    st.pyplot(plt)
except Exception as e:
    st.error(f"Error generating tear sheet: {e}")

st.write("The Pyfolio tear sheet provides a visual overview of the backtested strategy's performance, offering metrics like cumulative returns, drawdown, and risk-adjusted returns.")
st.markdown('---')

# Conclusion Section
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
<h2 style="font-size: 24px; font-weight: bold;">Machine Learning Model used:</h2>
<h2 style="font-size: 18px; font-weight: bold; color: red;">-  Multiple Linear Regression</h2>
 
<h2 style="font-size: 24px; font-weight: bold;">Training Period:</h2>
<h2 style="font-size: 18px; font-weight: bold; color: red;">- {train_start_date} to {train_end_date}</h2>

<h2 style="font-size: 24px; font-weight: bold;">Backtesting Period:</h2>
<h2 style="font-size: 18px; font-weight: bold; color: red;"> - {backtest_start_date} to {backtest_end_date}</h2>

<h2 style="font-size: 24px; font-weight: bold;">Features / Technical Indicators used:</h2>
<h3 style="font-size: 18px; font-weight: bold; color: red;">  - ADX (Average Directional Index)</h3>
<h3 style="font-size: 18px; font-weight: bold; color: red;">  - Volatility</h3>
<h3 style="font-size: 18px; font-weight: bold; color: red;">  - Correlation between price and SMA</h3>
<h3 style="font-size: 18px; font-weight: bold; color: red;">  - RSI (Relative Strength Index)</h3>
<h2 style="font-size: 24px; font-weight: bold;">Model accuracy:</h2>
<h2 style="font-size: 48px; font-weight: bold; color: red;">  {accuracy:.2f}%</h2>

<h2 style="font-size: 24px; font-weight: bold;">Cumulative Returns during the Backtesting Period:</h2>
<h2 style="font-size: 88px; font-weight: bold; color: red;">  {cumulative_returns}</h2>
""", unsafe_allow_html=True)

st.write("This conclusion summarizes the model used, training and backtesting periods, the technical indicators incorporated into the model, its accuracy, and the cumulative returns achieved during backtesting.")
st.markdown('---')

# Footer
st.markdown("---")
st.write("Created by Dr. Alvin Ang")
