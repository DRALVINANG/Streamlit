# --------------------------------------------------------------------------------------
# Step 1: Install and Import Libraries
# --------------------------------------------------------------------------------------

import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import streamlit as st

st.set_option('client.showErrorDetails', False)

# --------------------------------------------------------------------------------------
# Step 2: Load and Prepare Data
# --------------------------------------------------------------------------------------

@st.cache
def load_data(ticker, start_date):
    # Download the stock data using yfinance for the given ticker symbol
    data1 = yf.download(ticker, start=start_date)
    
    # Dropping the multi-level column in case of 'Adj Close' and setting it to a single level
    data1.columns = data1.columns.droplevel(level=1)
    
    # Convert the 'Date' index to datetime format (yfinance already returns data with datetime index)
    data1.index = pd.to_datetime(data1.index)
    
    return data1

# --------------------------------------------------------------------------------------
# Step 3: Apply Technical Indicators
# --------------------------------------------------------------------------------------

def apply_technical_indicators(df):
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
    
    return df

# --------------------------------------------------------------------------------------
# Step 4: Streamlit App Interface
# --------------------------------------------------------------------------------------

st.title("Stock Price and Technical Indicators")

# Sidebar for entering the ticker symbol and start date
ticker = st.sidebar.text_input("Enter Stock Ticker Symbol", value='TSLA')
start_date = st.sidebar.date_input("Select Start Date", value=pd.to_datetime("2020-01-01"))

# Load the data based on user input
if ticker:
    df = load_data(ticker, start_date)
    df = apply_technical_indicators(df)

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
        st.subheader(f"{ticker} Stock Price and 5-Period SMA")
        st.line_chart(df[['Close', 'SMA_5']])

    elif indicator == 'Close Price, SMA, and EMA':
        st.subheader(f"{ticker} Stock Price, 5-Period SMA, and 5-Period EMA")
        st.line_chart(df[['Close', 'SMA_5', 'EMA_5']])

    elif indicator == 'RSI':
        st.subheader(f"{ticker} Stock Price and 5-Period RSI")
        st.line_chart(df[['RSI_5']])

    elif indicator == 'Bollinger Bands':
        st.subheader(f"{ticker} Stock Price with Bollinger Bands")
        st.line_chart(df[['Close', 'Lower', 'Mid', 'Upper']])

    elif indicator == 'ADX':
        st.subheader(f"{ticker} Stock Price and 14-Period ADX")
        st.line_chart(df[['ADX_14']])

    elif indicator == 'MACD':
        st.subheader(f"{ticker} Stock Price - MACD, Signal Line, and Histogram")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['MACD'], label='MACD', color='blue', linewidth=2)
        ax.set_title(f"{ticker} Stock Price - MACD, Signal Line, and Histogram")
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
            title=f"{ticker} Stock Price",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
        )
        st.plotly_chart(fig)

    elif chart_type == 'Heikin Ashi':
        ha_df = ta.ha(open_=df.Open, high=df.High, low=df.Low, close=df.Close)
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=ha_df.HA_open, high=ha_df.HA_high, low=ha_df.HA_low, close=ha_df.HA_close)])
        fig.update_layout(
            title=f"{ticker} Stock Price",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
        )
        st.plotly_chart(fig)

# --------------------------------------------------------------------------------------
# Step 6: End of Analysis
# --------------------------------------------------------------------------------------
st.write("Created by Dr. Alvin Ang.")

