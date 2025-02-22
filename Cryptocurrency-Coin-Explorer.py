import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# List of cryptocurrency coins with their corresponding GitHub CSV file links
crypto_coin = [
    {
        "name": "Bitcoin",
        "ticker": 'BTC',
        "price": 35870.86,
        "24h_change": 1.48,
        "volume": 670552492185,
        "supply": 18721781,
        "category": 'CURRENCY',
        "csv_url": "https://raw.githubusercontent.com/DRALVINANG/Streamlit/refs/heads/main/datasets/btc_2024.csv"  # Corrected URL
    },
    {
        "name": "Ethereum",
        "ticker": 'ETH',
        "price": 2422.80,
        "24h_change": 0.05,
        "volume": 281073942766,
        "supply": 116083174,
        "category": 'PLATFORM',
        "csv_url": "https://raw.githubusercontent.com/DRALVINANG/Streamlit/refs/heads/main/datasets/eth_2024.csv"
    },
    {
        "name": "Cardano",
        "ticker": 'ADA',
        "price": 1.63,
        "24h_change": 10.26,
        "volume": 51951358766,
        "supply": 31948309441,
        "category": 'PLATFORM',
        "csv_url": "https://raw.githubusercontent.com/DRALVINANG/Streamlit/refs/heads/main/datasets/ada_2024.csv"
    },
    {
        "name": "Binance Coin",
        "ticker": 'BNB',
        "price": 331.21,
        "24h_change": 6.08,
        "volume": 50829735875,
        "supply": 153432897,
        "category": 'EXCHANGE',
        "csv_url": "https://raw.githubusercontent.com/DRALVINANG/Streamlit/refs/heads/main/datasets/bnb_2024.csv"
    },
    {
        "name": "XRP",
        "ticker": 'XRP',
        "price": 5.00,
        "24h_change": 7.39,
        "volume": 40594034312,
        "supply": 46143602688,
        "category": 'CURRENCY',
        "csv_url": "https://raw.githubusercontent.com/DRALVINANG/Streamlit/refs/heads/main/datasets/xrp_2024.csv"
    },
    {
        "name": "Dogecoin",
        "ticker": 'DOGE',
        "price": 0.31,
        "24h_change": 2.11,
        "volume": 39593068555,
        "supply": 129813129789,
        "category": 'CURRENCY',
        "csv_url": "https://raw.githubusercontent.com/DRALVINANG/Streamlit/refs/heads/main/datasets/doge_2024.csv"
    }
]

# Streamlit app
st.markdown("<h1 style='color: red; font-weight: bold;'>Cryptocurrency Coin Explorer</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: brown;'>Explore the details and historical pricing of various cryptocurrencies.</p>", unsafe_allow_html=True)

# Select a Coin section
st.markdown("## **Select a Coin**")

# Create a dropdown menu with coin names
coin_names = [coin['name'] for coin in crypto_coin]
selected_coin_name = st.selectbox("Select a Coin", coin_names)

# Find the selected coin's details
selected_coin = next(coin for coin in crypto_coin if coin['name'] == selected_coin_name)

# Display the selected coin's details in a CSV preview format
st.subheader("Coin Details Preview")
coin_details_df = pd.DataFrame({
    "Name": [selected_coin['name']],
    "Ticker": [selected_coin['ticker']],
    "Price": [selected_coin['price']],
    "24h Change": [selected_coin['24h_change']],
    "Volume": [selected_coin['volume']],
    "Supply": [selected_coin['supply']],
    "Category": [selected_coin['category']]
})

# Fetch historical data for the selected coin from the GitHub CSV link
coin_data = pd.read_csv(selected_coin['csv_url'])

# Add Open, High, Low, Close prices to the preview
coin_details_df = pd.concat([coin_details_df, coin_data[['Open', 'High', 'Low', 'Close']].tail(5)], axis=1)

st.write(coin_details_df)

# Download button for CSV with date index
coin_data.reset_index(inplace=True)  # Reset index to include date in the DataFrame
combined_df = pd.concat([coin_details_df, coin_data[['Date', 'Open', 'High', 'Low', 'Close']].tail(5)], axis=1)
csv_data = combined_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name=f"{selected_coin['ticker']}_details.csv",
    mime="text/csv"
)

# Horizontal line separator
st.markdown("---")

# Select a Condition section
st.markdown("## **Select a Condition**")

# Create a dropdown menu with conditions
conditions = ["Category is PLATFORM", "24h Change > 5%"]
selected_condition = st.selectbox("Select a Condition", conditions)

# Evaluate the condition and display the result
if selected_condition == "Category is PLATFORM":
    result = 'PLATFORM' in selected_coin['category']
elif selected_condition == "24h Change > 5%":
    result = selected_coin['24h_change'] > 5

st.write(f"**Condition:** {selected_condition}")
st.write(f"**Result:** {result}")

# Provide recommendation based on the condition
if selected_condition == "24h Change > 5%":
    if result:
        st.write(f"**Recommendation:** Buy {selected_coin['name']}! It's volatile enough.")
    else:
        st.write(f"**Recommendation:** Do not buy {selected_coin['name']}! Not volatile enough.")

# Horizontal line separator before the chart
st.markdown("---")

# Create candlestick chart
candlestick = go.Figure(data=[go.Candlestick(
    x=coin_data['Date'],
    open=coin_data["Open"],
    high=coin_data["High"],
    low=coin_data["Low"],
    close=coin_data["Close"]
)])

# Customize chart layout
candlestick.update_layout(
    title={
        'text': f"{selected_coin['ticker']} Candlestick Chart",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)

candlestick.update_yaxes(title_text=f"{selected_coin['ticker']} Price")

# Display the chart in Streamlit
st.plotly_chart(candlestick)

# Brief description of the coin's history
coin_history = {
    "BTC": "Bitcoin is the first decentralized cryptocurrency, created in 2009 by an anonymous person (or group) known as Satoshi Nakamoto. It allows peer-to-peer transactions without the need for intermediaries, relying on blockchain technology to secure and verify transactions.",
    "ETH": "Ethereum is a decentralized platform that enables developers to build and deploy smart contracts and decentralized applications (dApps). Launched in 2015, it introduced the concept of programmable money and is the second-largest cryptocurrency by market capitalization.",
    "ADA": "Cardano is a blockchain platform that aims to provide a more secure and scalable infrastructure for the development of dApps and smart contracts. Launched in 2017, it is known for its research-driven approach and uses a proof-of-stake consensus mechanism.",
    "BNB": "Binance Coin is the native cryptocurrency of the Binance exchange, launched in 2017. It is used for trading fee discounts, token sales, and various applications within the Binance ecosystem.",
    "XRP": "XRP is the native digital asset of the Ripple network, designed for fast and low-cost international money transfers. Launched in 2012, it aims to facilitate cross-border payments and is used by several financial institutions.",
    "DOGE": "Dogecoin started as a meme cryptocurrency in 2013, featuring the Shiba Inu dog from the 'Doge' meme. Initially created as a joke, it has gained a large community and is used for tipping and charitable donations."
}

# Display the coin's history
st.subheader("Coin History")
st.write(coin_history[selected_coin['ticker']])

