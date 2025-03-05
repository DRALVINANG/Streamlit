import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Define a list of stock tickers and their respective GitHub CSV URLs
stock_data_urls = {
    "TSLA": "https://raw.githubusercontent.com/DRALVINANG/Streamlit/refs/heads/main/datasets/tsla_2024.csv",
    "AAPL": "https://raw.githubusercontent.com/DRALVINANG/Streamlit/refs/heads/main/datasets/aapl_2024.csv",
    "AMZN": "https://raw.githubusercontent.com/DRALVINANG/Streamlit/refs/heads/main/datasets/amzn_2024.csv",
    "MSFT": "https://raw.githubusercontent.com/DRALVINANG/Streamlit/refs/heads/main/datasets/msft_2024.csv",
    "NVDA": "https://raw.githubusercontent.com/DRALVINANG/Streamlit/refs/heads/main/datasets/nvda_2024.csv",
    "INTC": "https://raw.githubusercontent.com/DRALVINANG/Streamlit/refs/heads/main/datasets/intc_2024.csv"
}

# Create a Streamlit app
st.title('Predicting the Close Price of a Tech Stock')
st.write('## Applying Linear Regression (LR) to predict Today\'s close price using Yesterday\'s high price')
st.write('This app analyzes the relationship between the high and close prices of a selected tech stock and uses linear regression to predict the close price based on a hypothetical high price.')

# Add a selectbox for the user to choose a stock
selected_ticker = st.selectbox('Select a stock', list(stock_data_urls.keys()))

# Add date input widgets with default values
start_date = st.date_input('Start date', value=pd.to_datetime('2020-01-01'))
end_date = st.date_input('End date', value=pd.to_datetime('2021-12-31'))

# Fetch the stock data from the GitHub CSV link
stock_data_url = stock_data_urls[selected_ticker]
df = pd.read_csv(stock_data_url)

# Ensure the 'Date' column is converted to datetime if it exists
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Shift the High price to yesterday's price
X = df['High'].shift(+1)  # yesterday's High
y = df['Close']           # today's Close

# Create a new dataframe with the shifted High price and Close price
df2 = pd.DataFrame({'High' : X, 'Close' : y})

# Drop the rows with NaN values
df2 = df2.dropna()

# Plot the Close price vs Date
st.subheader('Close Price vs Date')
df1 = df.reset_index()
df1 = df1[['Date', 'Close']]
df1.set_index('Date', inplace=True)
plt.figure(figsize=(12, 6))
plt.plot(df1, 'b')
plt.plot(df1, 'ro')
plt.grid(True)
plt.title('Close Price Representation')
plt.xlabel('Date')
plt.ylabel('$ Close Price')
st.pyplot(plt)

# Plot the High price vs Close price
st.subheader('Close Price vs High Price')
plt.figure(figsize=(12, 6))
plt.scatter(df2['High'], df2['Close'], color='blue', alpha=0.7)
plt.title('Scatter Plot of High vs Close Price')
plt.xlabel("Yesterday's High Price ($)")
plt.ylabel("Today's Close Price ($)")
plt.grid(True)
st.pyplot(plt)

# Linear Regression
st.write('---')  # Add a horizontal line to separate the segments
st.subheader('Linear Regression')
st.write('In this section, we apply linear regression to predict the close price based on the high price. The linear regression equation is in the form of y = mx + c, where m is the slope and c is the intercept.')

# Use the shifted High price and Close price for linear regression
X = df2['High']
y = df2['Close']

split = int(0.8*len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Convert Series to DataFrames
X_train = X_train.to_frame()
y_train = y_train.to_frame()
X_test = X_test.to_frame()
y_test = y_test.to_frame()

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display the equation
m = model.coef_
c = model.intercept_
equation = f'y = {m.item():.2f}x + {c.item():.2f}'
st.write(f'The equation is: {equation}')

# User input for hypothetical High price
high_price = st.number_input('Enter a hypothetical High price (for Today):', value=0.0)

# Predict the Close price
predicted_close_price = model.predict([[high_price]])
st.write(f'The predicted Close price (for tomorrow) is: ${predicted_close_price.item():.2f}')

# Footer
st.write('---')  # Add a horizontal line to separate the segments
st.write('**Created by Dr. Alvin Ang**', style={'font-style': 'italic'})
st.write('---')  # Add a horizontal line to separate the segments
