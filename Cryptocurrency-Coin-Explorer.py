import streamlit as st

# List of cryptocurrency coins
crypto_coin = [
    {
        "name": "Bitcoin",
        "ticker": 'BTC',
        "price": 35870.86,
        "24h_change": 1.48,
        "volume": 670552492185,
        "supply": 18721781,
        "category": 'CURRENCY'
    },
    {
        "name": "Ethereum",
        "ticker": 'ETH',
        "price": 2422.80,
        "24h_change": 0.05,
        "volume": 281073942766,
        "supply": 116083174,
        "category": 'PLATFORM'
    },
    {
        "name": "Cardano",
        "ticker": 'ADA',
        "price": 1.63,
        "24h_change": 10.26,
        "volume": 51951358766,
        "supply": 31948309441,
        "category": 'PLATFORM'
    },
    {
        "name": "Binance Coin",
        "ticker": 'BNB',
        "price": 331.21,
        "24h_change": 6.08,
        "volume": 50829735875,
        "supply": 153432897,
        "category": 'EXCHANGE'
    },
    {
        "name": "XRP",
        "ticker": 'XRP',
        "price": 5.00,
        "24h_change": 7.39,
        "volume": 40594034312,
        "supply": 46143602688,
        "category": 'CURRENCY'
    },
    {
        "name": "Dogecoin",
        "ticker": 'DOGE',
        "price": 0.31,
        "24h_change": 2.11,
        "volume": 39593068555,
        "supply": 129813129789,
        "category": 'CURRENCY'
    }
]

# Streamlit app
st.title("Cryptocurrency Coin Explorer")

# Select a Coin section
st.markdown("## **Select a Coin**")

# Create a dropdown menu with coin names
coin_names = [coin['name'] for coin in crypto_coin]
selected_coin_name = st.selectbox("Select a Coin", coin_names)

# Find the selected coin's details
selected_coin = next(coin for coin in crypto_coin if coin['name'] == selected_coin_name)

# Display the selected coin's details
st.write(f"**Name:** {selected_coin['name']}")
st.write(f"**Ticker:** {selected_coin['ticker']}")
st.write(f"**Price:** ${selected_coin['price']:.2f}")
st.write(f"**24h Change:** {selected_coin['24h_change']:.2f}%")
st.write(f"**Volume:** ${selected_coin['volume']:,.2f}")
st.write(f"**Supply:** {selected_coin['supply']:,.0f}")
st.write(f"**Category:** {selected_coin['category']}")

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
# Add more conditions as needed

st.write(f"**Condition:** {selected_condition}")
st.write(f"**Result:** {result}")

# Provide recommendation based on the condition
if selected_condition == "24h Change > 5%":
    if result:
        st.write(f"**Recommendation:** Buy {selected_coin['name']}! It's volatile enough.")
    else:
        st.write(f"**Recommendation:** Do not buy {selected_coin['name']}! Not volatile enough.")
