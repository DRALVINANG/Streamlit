import streamlit as st
import yfinance as yf
import pandas as pd

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    return stock

def display_stock_info(stock):
    info = stock.info
    df1 = pd.DataFrame(info.items(), columns=["Key", "Value"])
    st.write("### Stock Information")
    st.dataframe(df1)

def display_stock_actions(stock):
    st.write("### Stock Actions (Dividends & Splits)")
    st.dataframe(stock.actions)

def display_stock_financials(stock):
    st.write("### Stock Financials")
    st.dataframe(stock.financials)

def display_stock_quarterly_financials(stock):
    st.write("### Stock Quarterly Financials")
    st.dataframe(stock.quarterly_financials)

def display_major_holders(stock):
    st.write("### Major Holders")
    st.dataframe(stock.major_holders)

def display_institutional_holders(stock):
    st.write("### Institutional Holders")
    st.dataframe(stock.institutional_holders)

def display_balance_sheet(stock):
    st.write("### Balance Sheet")
    st.dataframe(stock.balance_sheet)

def display_quarterly_balance_sheet(stock):
    st.write("### Quarterly Balance Sheet")
    st.dataframe(stock.quarterly_balance_sheet)

def display_cashflow(stock):
    st.write("### Cashflow")
    st.dataframe(stock.cashflow)

def display_quarterly_cashflow(stock):
    st.write("### Quarterly Cashflow")
    st.dataframe(stock.quarterly_cashflow)

def display_earnings(stock):
    st.write("### Earnings")
    st.dataframe(stock.earnings)

def display_quarterly_earnings(stock):
    st.write("### Quarterly Earnings")
    st.dataframe(stock.quarterly_earnings)

def display_sustainability(stock):
    st.write("### Sustainability")
    st.dataframe(stock.sustainability)

def display_recommendations(stock):
    st.write("### Analysts Recommendations")
    st.dataframe(stock.recommendations)

def display_calendar(stock):
    st.write("### Calendar (Next Events, Earnings, etc.)")
    st.dataframe(stock.calendar)

def display_isin(stock):
    st.write("### ISIN (International Securities Identification Number)")
    st.write(stock.isin)

def display_options(stock):
    st.write("### Options Expirations")
    st.write(stock.options)

def display_news(stock):
    st.write("### News")
    st.write(stock.news)

# Streamlit App layout
def main():
    st.title('Stock Information Viewer using YFinance')

    # Input for the user to enter the stock ticker
    ticker = st.text_input('Enter Stock Ticker (e.g. D05.SI, AAPL)', 'D05.SI')
    
    if ticker:
        stock = get_stock_data(ticker)
        
        # Display stock information
        display_stock_info(stock)
        display_stock_actions(stock)
        display_stock_financials(stock)
        display_stock_quarterly_financials(stock)
        display_major_holders(stock)
        display_institutional_holders(stock)
        display_balance_sheet(stock)
        display_quarterly_balance_sheet(stock)
        display_cashflow(stock)
        display_quarterly_cashflow(stock)
        display_earnings(stock)
        display_quarterly_earnings(stock)
        display_sustainability(stock)
        display_recommendations(stock)
        display_calendar(stock)
        display_isin(stock)
        display_options(stock)
        display_news(stock)

if __name__ == '__main__':
    main()
