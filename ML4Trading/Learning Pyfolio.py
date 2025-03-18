import streamlit as st
import yfinance as yf
import pandas as pd
import pyfolio as pf
import matplotlib.pyplot as plt

# Set the style for matplotlib
plt.style.use('fivethirtyeight')

# Streamlit app header
st.title("Stock Analysis with Pyfolio")
st.write("This app provides stock analysis including the stock price, daily returns, cumulative returns, performance statistics, and a Pyfolio tear sheet to evaluate the performance of a stock.")

# Input for the stock ticker
ticker = st.text_input("Enter Stock Ticker", value="D05.SI")

# Define the start date
start_date = '2023-01-01'

# Download stock historical data based on ticker input
@st.cache_data
def get_stock_data(ticker, start_date):
    # Download data from Yahoo Finance
    data = yf.download(ticker, start=start_date)
    
    # Flatten multi-level column headers if necessary
    data.columns = data.columns.droplevel(level=1)  # Flatten the column header
    return data

# Fetch the data for the given ticker
stock_history = get_stock_data(ticker, start_date)

# Display stock price chart
st.subheader(f'{ticker} Stock Price')
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(stock_history['Close'], label=f'{ticker} Close Price')
ax.set_title(f'{ticker} Stock Price')
ax.set_ylabel('Price (SGD)')
ax.set_xlabel('Date')
ax.legend()
st.pyplot(fig)

# Description below the stock price chart
st.write(f"The chart above shows the closing price of {ticker} over time, allowing you to visualize its historical performance.")

# Add a horizontal line to separate sections
st.markdown("---")

# Calculate daily returns
stock_returns = stock_history['Close'].pct_change().dropna()

# Display daily returns chart
st.subheader(f'{ticker} Daily Returns')
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(stock_returns.index, stock_returns, label=f'{ticker} Daily Returns')
ax.set_title(f'{ticker} Daily Returns')
ax.set_ylabel('Returns')
ax.set_xlabel('Date')
ax.legend()
st.pyplot(fig)

# Description below the daily returns chart
st.write(f"The bar chart above represents the daily percentage returns of {ticker}. Each bar shows the daily change in the stock price.")

# Add a horizontal line to separate sections
st.markdown("---")

# Calculate cumulative returns using Pyfolio and display plot
st.subheader(f'{ticker} Cumulative Returns')
fig, ax = plt.subplots(figsize=(10, 5))
pf.timeseries.cum_returns(stock_returns).plot(ax=ax)
ax.set_title(f'Cumulative Returns for {ticker}')
st.pyplot(fig)

# Description below the cumulative returns chart
st.write(f"The line chart above shows the cumulative returns for {ticker}. It helps to visualize how the stock's value accumulates over time.")

# Add a horizontal line to separate sections
st.markdown("---")

# Display performance statistics (check if returns data is not empty)
st.subheader(f'{ticker} Performance Statistics')
if not stock_returns.empty:
    try:
        perf_stats = pf.timeseries.perf_stats(stock_returns)
        
        # Display each performance statistic line by line
        for stat_name, stat_value in perf_stats.items():
            st.write(f"{stat_name}: {stat_value:.4f}")  # Display each stat with 4 decimal points
    except Exception as e:
        st.error(f"Error calculating performance stats: {e}")
else:
    st.error("Invalid stock returns data. Please check the ticker and try again.")

# Add a horizontal line to separate sections
st.markdown("---")

# Display Pyfolio Simple Tear Sheet
st.subheader(f'{ticker} Pyfolio Tear Sheet')
if not stock_returns.empty:
    try:
        # Remove the 'fig' argument from create_simple_tear_sheet as it does not accept it
        pf.create_simple_tear_sheet(stock_returns)
        st.pyplot(plt)  # Displaying the tear sheet using matplotlib
    except Exception as e:
        st.error(f"Error generating tear sheet: {e}")
else:
    st.error("Invalid stock returns data. Please check the ticker and try again.")

# Add a horizontal line to separate sections
st.markdown("---")

# Display "Created by Dr. Alvin Ang" at the bottom
st.write("Created by Dr. Alvin Ang")
