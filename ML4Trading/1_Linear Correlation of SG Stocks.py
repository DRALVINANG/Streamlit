# -------------------------------------------------------------------------------------
# Step 1: Pip Install and Import Libraries
# -------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression
import datetime

# -------------------------------------------------------------------------------------
# Step 2: Set Streamlit Page Config
# -------------------------------------------------------------------------------------
st.set_page_config(page_title="Stock Price Correlation", layout="wide")

# Title with an icon
st.title("📈 Stock Price Correlation Viewer")
st.markdown("""
This interactive app lets you compare the correlation between stock prices of different companies. You can select companies (e.g., Singapore banks like DBS, UOB, or telcos like Singtel, Starhub), specify a date range, and visually explore their linear relationship through scatter plots and linear regression models.
""")

# -------------------------------------------------------------------------------------
# Step 3: Sidebar for Interactivity
# -------------------------------------------------------------------------------------
st.sidebar.header("Select Tickers and Date Range")

# Tickers and their company names (Updated)
tickers_dict = {
    'D05.SI': 'DBS Bank',
    'U11.SI': 'UOB Bank',
    'S68.SI': 'Singtel',
    'Z74.SI': 'Starhub',
    'A17U.SI': 'Mapletree Industrial Trust',
    'STAN.L': 'Standard Chartered',
    'S63.SI': 'ST Engineering'  # Replaced JPMorgan Chase with ST Engineering
}

# Create a list of tickers with corresponding company names
tickers = list(tickers_dict.keys())

# Dropdown menu for selecting stock tickers with company names
selected_ticker1 = st.sidebar.selectbox("Select the first stock", tickers, format_func=lambda x: f"{x} - {tickers_dict[x]}")
selected_ticker2 = st.sidebar.selectbox("Select the second stock", [t for t in tickers if t != selected_ticker1], format_func=lambda x: f"{x} - {tickers_dict[x]}")

# Date Range Slider
start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# -------------------------------------------------------------------------------------
# Step 4: Download Data
# -------------------------------------------------------------------------------------
@st.cache_data
def download_data(ticker1, ticker2, start, end):
    data1 = yf.download(ticker1, start=start, end=end)
    data2 = yf.download(ticker2, start=start, end=end)

    # Dropping the multi-level column index after downloading the data
    data1.columns = data1.columns.droplevel(level=1)
    data2.columns = data2.columns.droplevel(level=1)

    return data1, data2

# Show loading bar
with st.spinner('Loading data...'):
    data1, data2 = download_data(selected_ticker1, selected_ticker2, start_date, end_date)

# -------------------------------------------------------------------------------------
# Step 5: Stock Data Preview
# -------------------------------------------------------------------------------------
st.header("📊 Stock Data Preview")
st.subheader(f"Stock Data for {tickers_dict[selected_ticker1]} and {tickers_dict[selected_ticker2]}")

# Displaying data preview for both stocks
st.write("Preview of stock data for the selected companies:")
st.write(data1[['Close']].tail(), f"{tickers_dict[selected_ticker1]}")
st.write(data2[['Close']].tail(), f"{tickers_dict[selected_ticker2]}")

st.markdown("---")

# -------------------------------------------------------------------------------------
# Step 6: Merging Data
# -------------------------------------------------------------------------------------
st.header("🔗 Merging Data")
st.subheader(f"Aligning {tickers_dict[selected_ticker1]} and {tickers_dict[selected_ticker2]} Close Prices")

# Merging the data based on the date index
data = pd.merge(data1['Close'], data2['Close'], left_index=True, right_index=True, how='inner')  # Ensure we only keep dates with data for both

# Check if merged data is empty
if data.empty:
    st.error(f"No overlapping data found for {tickers_dict[selected_ticker1]} and {tickers_dict[selected_ticker2]} in the selected date range.")
else:
    # Rename columns
    data.rename(columns={'Close_x': f'{tickers_dict[selected_ticker1]} Close Price', 'Close_y': f'{tickers_dict[selected_ticker2]} Close Price'}, inplace=True)

    # Show merged data preview
    st.write("Merged Data Preview:", data.tail())

    # Check for missing values in the data
    missing_data = data.isna().sum()
    st.write("Missing Data in Columns:", missing_data)

    st.markdown("---")

# -------------------------------------------------------------------------------------
# Step 7: Scatter Plot
# -------------------------------------------------------------------------------------
st.header("🔍 Scatter Plot")
st.subheader(f"Scatter Plot of {tickers_dict[selected_ticker1]} vs {tickers_dict[selected_ticker2]} Close Prices")

# Scatter plot
plt.figure(figsize=(12, 5))
plt.scatter(data[f'{tickers_dict[selected_ticker1]} Close Price'], data[f'{tickers_dict[selected_ticker2]} Close Price'])
plt.xlabel(f'{tickers_dict[selected_ticker1]} Price')
plt.ylabel(f'{tickers_dict[selected_ticker2]} Price')
plt.title(f"Scatter plot: {tickers_dict[selected_ticker1]} vs {tickers_dict[selected_ticker2]}")
st.pyplot(plt)

st.markdown("""
The scatter plot above shows the relationship between the two stocks over the selected date range. A positive correlation means that the prices move in the same direction, while a negative correlation indicates an inverse relationship.
""")

st.markdown("---")

# -------------------------------------------------------------------------------------
# Step 8: Linear Regression Model
# -------------------------------------------------------------------------------------
st.header("📉 Linear Regression")
st.subheader(f"Linear Regression Model between {tickers_dict[selected_ticker1]} and {tickers_dict[selected_ticker2]}")

# Assigning variables for regression
X = data[[f'{tickers_dict[selected_ticker1]} Close Price']]  # Independent variable (DBS)
y = data[f'{tickers_dict[selected_ticker2]} Close Price']   # Dependent variable (UOB)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Get the coefficients
intercept = model.intercept_
slope = model.coef_[0]
r_squared = model.score(X, y)

# Show model summary
st.write(f"*Model Equation:*")
st.write(f"y = {intercept:.2f} + {slope:.2f} * x")
st.write(f"*R-squared:* {r_squared:.4f}")

st.markdown("""
The linear regression model above shows the relationship between the two stocks. The equation of the line can be derived from the model's coefficients. The R-squared value indicates how well the model fits the data (values closer to 1 imply a strong relationship).
""")

st.markdown("---")

# -------------------------------------------------------------------------------------
# Step 9: Plot Regression Line
# -------------------------------------------------------------------------------------
st.header("📈 Linear Regression Line")
st.subheader(f"Regression Line of {tickers_dict[selected_ticker1]} vs {tickers_dict[selected_ticker2]}")

# Regression plot
plt.figure(figsize=(12, 5))
plt.scatter(data[f'{tickers_dict[selected_ticker1]} Close Price'], data[f'{tickers_dict[selected_ticker2]} Close Price'])
plt.plot(data[f'{tickers_dict[selected_ticker1]} Close Price'], intercept + slope * data[f'{tickers_dict[selected_ticker1]} Close Price'], color='red')
plt.xlabel(f'{tickers_dict[selected_ticker1]} Price')
plt.ylabel(f'{tickers_dict[selected_ticker2]} Price')
plt.title(f"Linear Regression: {tickers_dict[selected_ticker1]} vs {tickers_dict[selected_ticker2]}")
st.pyplot(plt)

st.markdown("""
The red line represents the linear regression model, which predicts the price of one stock based on the price of the other.
""")

st.markdown("---")

# -------------------------------------------------------------------------------------
# Step 10: End Message
# -------------------------------------------------------------------------------------
st.markdown("""
Created by Dr. Alvin Ang
""")
