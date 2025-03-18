# --------------------------------------------------------------------------------------
# Step 1: Install and Import Libraries
# --------------------------------------------------------------------------------------

import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------------------
# Step 2: Load and Prepare Data
# --------------------------------------------------------------------------------------

@st.cache
def load_data():
    # Load the dataset
    df = pd.read_csv('https://gist.githubusercontent.com/DRALVINANG/5821855d6bcce977fc7f7638bb7ea9a3/raw/9d5bf33a581bf81a8319baf9d677eef309a2d7e9/TSLA%2520Stock%2520Price%2520(2020).csv')

    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Set the 'Date' column as the index
    df.set_index('Date', inplace=True)
    
    return df

df = load_data()

# --------------------------------------------------------------------------------------
# Step 3: Apply Technical Indicators
# --------------------------------------------------------------------------------------

# Simple Moving Average (SMA)
df['SMA_5'] = ta.sma(close=df.Close, length=5)

# Exponential Moving Average (EMA)
df['EMA_5'] = ta.ema(close=df['Close'], length=5)

# Relative Strength Index (RSI)
df['RSI_5'] = ta.rsi(close=df.Close, length=5)

# Bollinger Bands
bb = ta.bbands(df.Close, length=14, std=2)
bb.drop(['BBB_14_2.0', 'BBP_14_2.0'], axis=1, inplace=True)
bb.columns = ['Lower', 'Mid', 'Upper']
df = df.join(bb)

# Average Directional Index (ADX)
adx = ta.adx(df.High, df.Low, df.Close, length=14)
df = df.join(adx)
df.drop(['DMP_14', 'DMN_14'], axis=1, inplace=True)

# Moving Average Convergence Divergence (MACD)
macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
df['MACD'] = macd['MACD_12_26_9']

# --------------------------------------------------------------------------------------
# Step 4: Streamlit App Interface
# --------------------------------------------------------------------------------------

st.title("Tesla Stock Price and Technical Indicators (2020)")

# Sidebar for selecting the indicator to visualize
indicator = st.sidebar.selectbox("Select Indicator", [
    'Close Price and 5-Period SMA', 
    'Close Price, SMA, and EMA', 
    'RSI', 
    'Bollinger Bands', 
    'ADX', 
    'MACD'
])

# Display selected chart
if indicator == 'Close Price and 5-Period SMA':
    st.subheader("Tesla Stock Price and 5-Period SMA (2020)")
    st.line_chart(df[['Close', 'SMA_5']])

elif indicator == 'Close Price, SMA, and EMA':
    st.subheader("Tesla Stock Price, 5-Period SMA, and 5-Period EMA (2020)")
    st.line_chart(df[['Close', 'SMA_5', 'EMA_5']])

elif indicator == 'RSI':
    st.subheader("Tesla Stock Price and 5-Period RSI (2020)")
    st.line_chart(df[['RSI_5']])

elif indicator == 'Bollinger Bands':
    st.subheader("Tesla Stock Price with Bollinger Bands (2020)")
    st.line_chart(df[['Close', 'Lower', 'Mid', 'Upper']])

elif indicator == 'ADX':
    st.subheader("Tesla Stock Price and 14-Period ADX (2020)")
    st.line_chart(df[['ADX_14']])

elif indicator == 'MACD':
    st.subheader("Tesla Stock Price - MACD, Signal Line, and Histogram (2020)")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(macd.index, macd['MACD_12_26_9'], label='MACD', color='blue', linewidth=2)
    ax.plot(macd.index, macd['MACDs_12_26_9'], label='Signal Line', color='red', linewidth=2)
    ax.bar(macd.index, macd['MACDh_12_26_9'], label='Histogram', color='gray', alpha=0.5)
    ax.set_title("Tesla Stock Price - MACD, Signal Line, and Histogram (2020)")
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    st.pyplot(fig)

# --------------------------------------------------------------------------------------
# Step 5: Plot Candlestick Charts
# --------------------------------------------------------------------------------------

chart_type = st.sidebar.radio("Choose Candlestick Chart", ['Normal Candlestick', 'Heikin Ashi'])

if chart_type == 'Normal Candlestick':
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close)])
    fig.update_layout(
        title="Tesla Stock Price (2020)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
    )
    st.plotly_chart(fig)

elif chart_type == 'Heikin Ashi':
    ha_df = ta.ha(open_=df.Open, high=df.High, low=df.Low, close=df.Close)
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=ha_df.HA_open, high=ha_df.HA_high, low=ha_df.HA_low, close=ha_df.HA_close)])
    fig.update_layout(
        title="Tesla Stock Price (2020)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
    )
    st.plotly_chart(fig)

# --------------------------------------------------------------------------------------
# Step 6: End of Analysis
# --------------------------------------------------------------------------------------
st.write("End of Analysis.")

