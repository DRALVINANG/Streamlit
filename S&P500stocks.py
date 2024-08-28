import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import requests
from bs4 import BeautifulSoup

# Function to get S&P 500 tickers and company names
def get_sp500_tickers_and_names():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers_and_names = {}
    for row in table.find_all('tr')[1:]:
        cells = row.find_all('td')
        ticker = cells[0].text.strip()
        name = cells[1].text.strip()
        tickers_and_names[ticker] = name
    return tickers_and_names

# Function to get company information from Wikipedia
def get_company_info(ticker):
    company_name = sp500_data[ticker]
    search_url = f"https://en.wikipedia.org/wiki/{company_name.replace(' ', '_')}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    info_box = soup.find('table', {'class': 'infobox'})

    if not info_box:
        return None

    info = {}
    for row in info_box.find_all('tr'):
        header = row.find('th')
        cell = row.find('td')
        if header and cell:
            header_text = header.text.strip()
            cell_text = cell.text.strip()
            info[header_text] = cell_text

    return info

# Fetch S&P 500 tickers and company names
sp500_data = get_sp500_tickers_and_names()
tickers = list(sp500_data.keys())
companies = list(sp500_data.values())

# Create a dictionary for the dropdown display
ticker_company_pairs = [f"{ticker} - {name}" for ticker, name in sp500_data.items()]

# Set up Streamlit app
st.title("S&P 500 Stock Candlestick Chart")

# Dropdown for selecting the stock ticker
selected_option = st.selectbox("Select a stock ticker", ticker_company_pairs)

# Extract selected ticker from the dropdown selection
selected_ticker = selected_option.split(" - ")[0]

# Get stock data from Yahoo Finance
stock_data = yf.download(selected_ticker, start="2013-01-01", threads=False)

if stock_data.empty:
    st.error(f"No data found for ticker: {selected_ticker}")
else:
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
            'text': f"{selected_ticker} Share Price (2013-Today)",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    candlestick.update_yaxes(title_text=f"{selected_ticker} Close Price", tickprefix="$")

    # Display the chart in Streamlit
    st.plotly_chart(candlestick)

    # Fetch and display company information
    st.subheader("Company Information")

    try:
        company_info = get_company_info(selected_ticker)
        if company_info:
            st.write(f"**Company Name:** {sp500_data[selected_ticker]}")
            st.write(f"**Wikipedia Page:** [Link](https://en.wikipedia.org/wiki/{sp500_data[selected_ticker].replace(' ', '_')})")
            for key, value in company_info.items():
                st.write(f"**{key}:** {value}")
        else:
            st.write("No additional company information found.")
    except Exception as e:
        st.error(f"Error fetching company information: {e}")
