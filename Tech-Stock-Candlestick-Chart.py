import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

st.title("Stock Candlestick Chart")

# List of stock tickers
tickers = ["AAPL", "TSLA", "AMZN", "MSFT", "NVDA", "INTC"]

# Sidebar for selecting the stock
selected_stock = st.sidebar.selectbox("Select a stock", tickers)

# Get stock data from Yahoo Finance
stock_data = yf.download(selected_stock, start="2013-01-01")

# Create candlestick chart
candlestick = go.Figure(data=[go.Candlestick(
    x=stock_data.index,
    open=stock_data["Open"],
    high=stock_data["High"],
    low=stock_data["Low"],
    close=stock_data["Close"]
)])

# Customize chart layout
candlestick.update_xaxes(
    title_text="Date",
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

candlestick.update_layout(
    title={
        'text': f"{selected_stock} Share Price (2013-Today)",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)

candlestick.update_yaxes(title_text=f"{selected_stock} Close Price", tickprefix="$")

# Display the chart in Streamlit
st.plotly_chart(candlestick)
