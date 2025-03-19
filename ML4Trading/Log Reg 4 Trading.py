import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import pyfolio as pf
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Function to get target features
def get_target_features(data):
    data['PCT_CHANGE'] = data['Close'].pct_change()
    data['VOLATILITY'] = data['PCT_CHANGE'].rolling(14).std() * 100
    data['RSI'] = ta.rsi(data['Close'], timeperiod=14)
    data['ADX'] = ta.adx(data['High'], data['Low'], data['Close'], length=14)['ADX_14']
    data['SMA'] = ta.sma(data['Close'], timeperiod=14)
    data['CORR'] = data['Close'].rolling(window=14).corr(data['SMA'])
    data['Returns_4_Tmrw'] = data['Close'].pct_change().shift(-1)
    data['Actual_Signal'] = np.where(data['Returns_4_Tmrw'] > 0, 1, 0)
    data = data.dropna()
    return data['Actual_Signal'], data[['VOLATILITY', 'CORR', 'RSI', 'ADX']]

# Streamlit app starts here
st.title('Stock Prediction and Backtesting')

# Left Sidebar for ticker input
ticker = st.sidebar.text_input("Enter the Ticker Symbol", 'D05.SI')

# Download and plot stock data
data = yf.download(ticker, start='2022-01-01')
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

# Scaling features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Accuracy score as percentage
accuracy = metrics.accuracy_score(y_test, y_pred) * 100
st.subheader('Model Accuracy')

# Make the accuracy bold and larger
st.markdown(f"<h2 style='font-size: 36px; color: black; font-weight: bold;'>Accuracy: {accuracy:.2f}%</h2>", unsafe_allow_html=True)

# Description
st.write("The accuracy score represents how well the model predicts the direction of stock price movement based on the features used.")
st.markdown("---")

# Backtesting the strategy
df = yf.download(ticker, start='2016-01-01', end='2017-01-01')
df.columns = df.columns.droplevel(level=1)
df['PCT_CHANGE'] = df['Close'].pct_change()
df['VOLATILITY'] = df['PCT_CHANGE'].rolling(14).std() * 100
df['RSI'] = ta.rsi(df['Close'], timeperiod=14)
df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
df['SMA'] = ta.sma(df['Close'], timeperiod=14)
df['CORR'] = df['Close'].rolling(window=14).corr(df['SMA'])
df = df.dropna()

# Scale and predict using the trained model
df_scaled = sc.transform(df[['VOLATILITY', 'CORR', 'RSI', 'ADX']])
df['predicted_signal_4_tmrw'] = model.predict(df_scaled)

# Calculate strategy returns
df['strategy_returns'] = df['predicted_signal_4_tmrw'].shift(1) * df['PCT_CHANGE']
df.dropna(inplace=True)

# Display Pyfolio performance stats
st.subheader('Performance Stats')
perf_stats = pf.timeseries.perf_stats(df.strategy_returns)
st.write(perf_stats)  # Display the entire performance stats object

# Description
st.write("The performance stats provide insights into how the strategy performed over time, such as cumulative return, volatility, and maximum drawdown.")
st.markdown("---")

# Display Pyfolio tear sheet
st.subheader('Pyfolio Tear Sheet')
pf.create_simple_tear_sheet(df.strategy_returns)
st.pyplot(plt)

# Footer
st.markdown("---")
st.write("Created by Dr. Alvin Ang")

