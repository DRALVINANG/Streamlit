import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("Stock Candlestick Chart")

# List of stock tickers and their respective CSV URLs
stock_data_urls = {
    "AAPL": "https://raw.githubusercontent.com/DRALVINANG/Streamlit/refs/heads/main/datasets/aapl_2024.csv",
    "TSLA": "https://raw.githubusercontent.com/DRALVINANG/Streamlit/refs/heads/main/datasets/tsla_2024.csv",  # Example for TSLA
    "AMZN": "https://raw.githubusercontent.com/DRALVINANG/Streamlit/refs/heads/main/datasets/amzn_2024.csv",  # Example for AMZN
    "MSFT": "https://raw.githubusercontent.com/DRALVINANG/Streamlit/refs/heads/main/datasets/msft_2024.csv",  # Example for MSFT
    "NVDA": "https://raw.githubusercontent.com/DRALVINANG/Streamlit/refs/heads/main/datasets/nvda_2024.csv",  # Example for NVDA
    "INTC": "https://raw.githubusercontent.com/DRALVINANG/Streamlit/refs/heads/main/datasets/intc_2024.csv"   # Example for INTC
}

# Sidebar for selecting the stock
selected_stock = st.sidebar.selectbox("Select a stock", list(stock_data_urls.keys()))

# Get stock data from the GitHub CSV link
stock_data_url = stock_data_urls[selected_stock]
stock_data = pd.read_csv(stock_data_url)

# Ensure the 'Date' column is converted to datetime if it exists
if 'Date' in stock_data.columns:
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)

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
        'text': f"{selected_stock} Share Price (2013-2024)",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)

candlestick.update_yaxes(title_text=f"{selected_stock} Close Price", tickprefix="$")

# Display the chart in Streamlit
st.plotly_chart(candlestick)

