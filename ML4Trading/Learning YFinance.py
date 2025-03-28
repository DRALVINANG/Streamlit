import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_option('client.showErrorDetails', False)

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    return stock

def plot_candlestick_chart(stock):
    # Get historical data for the stock
    hist = stock.history(period="1y")  # Show 1 year of data by default
    fig = go.Figure(data=[go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close'],
        increasing_line_color='green', decreasing_line_color='red'
    )])

    fig.update_layout(
        title='Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig)

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
    st.write("### Latest News")
    news_data = stock.news
    
    if not news_data:
        st.write("No news available.")
        return
    
    for article in news_data:
        content = article.get('content', {})
        title = content.get('title', 'No Title')
        summary = content.get('summary', 'No summary available.')
        pub_date = content.get('pubDate', 'No date available.')
        link = content.get('clickThroughUrl', {}).get('url', '#')
        
        # Safely retrieve the thumbnail
        thumbnail = None
        if 'thumbnail' in content and content['thumbnail'] is not None:
            thumbnail = content['thumbnail'].get('originalUrl', None)
        
        st.write(f"**{title}**")
        st.write(f"Published: {pub_date}")
        st.write(f"Summary: {summary}")
        
        # Display the thumbnail if available
        if thumbnail:
            st.image(thumbnail, use_column_width=True)
        
        # Provide a clickable link to the full article
        st.markdown(f"[Read full article]({link})")
        st.write("\n---\n")

# Streamlit App layout
def main():
    st.title('Stock Information Viewer using YFinance')

    # Move the ticker input field to the sidebar
    ticker = st.sidebar.text_input('Enter Stock Ticker (e.g. D05.SI, AAPL)', 'D05.SI')
    
    if ticker:
        stock = get_stock_data(ticker)

        # Display the candlestick chart at the top
        plot_candlestick_chart(stock)
        
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

