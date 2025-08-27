import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# Step 1: Define Available Tech Stocks
# -----------------------------
tech_stocks = {
    'Apple': 'AAPL',
    'Microsoft': 'MSFT',
    'Google (Alphabet)': 'GOOGL',
    'Amazon': 'AMZN',
    'Tesla': 'TSLA',
    'Meta (Facebook)': 'META',
    'NVIDIA': 'NVDA'
}

# -----------------------------
# Streamlit App Interface
# -----------------------------

# Title of the app
st.title('Tech Stock RSI and Logistic Regression Analysis')

# Introduction to guide the user
st.write("""
    **How to Use this App:**
    
    1. **Choose a Stock**: Select a stock from the dropdown list in the sidebar. The app will download historical data for that stock.
    2. **Adjust RSI Settings**: Adjust the RSI length and the buy/sell RSI thresholds using the sliders. RSI (Relative Strength Index) is a momentum oscillator used to identify overbought or oversold conditions.
    3. **View Plots**: The app will show the RSI values, along with buy and sell thresholds, and a logistic regression curve that models the probability of buying or selling based on the RSI.
    4. **Model Evaluation**: The app will display the model's accuracy in percentage, compare predicted vs. actual buy/sell signals, and provide actionable recommendations.
    
    **Data Range Used**:
    The model is trained on historical data from **2018-01-01** to **2022-01-01**. This range ensures that there is enough data for analysis.
""")

# Dropdown for selecting the stock
selected_stock = st.sidebar.selectbox("Select a Tech Stock", list(tech_stocks.keys()))
ticker = tech_stocks[selected_stock]

# Date range for analysis - Extended for 3 years
start_date = '2018-01-01'
end_date = '2022-01-01'

# RSI length slider (limiting range for better model stability)
rsi_length = st.sidebar.slider("RSI Length", min_value=10, max_value=14, value=14)

# Buy/Sell RSI Thresholds slider
lower_threshold = st.sidebar.slider("Buy RSI Threshold", min_value=10, max_value=50, value=30)
upper_threshold = st.sidebar.slider("Sell RSI Threshold", min_value=50, max_value=90, value=70)

# -----------------------------
# Step 2: Download Data
# -----------------------------
# Download data for the selected stock
df = yf.download(ticker, start=start_date, end=end_date)

# Flatten multi-level columns (in case the update to yfinance requires it)
df.columns = df.columns.droplevel(level=1)

# -----------------------------
# Step 3: Calculate RSI
# -----------------------------
# Calculate RSI
df['RSI'] = ta.rsi(df['Close'], length=rsi_length)

# -----------------------------
# Step 4: Create the Signal using RSI
# -----------------------------
# Create Buy (1) and Sell (0) signals based on RSI thresholds
df['Signal'] = np.where(df['RSI'] < lower_threshold, 1,           # 1 means BUY
                        np.where(df['RSI'] > upper_threshold, 0,  # 0 means SELL
                                 np.nan))

# Drop rows with NaN values
df1 = df[['RSI', 'Signal']].dropna()

# Check if both classes are present
if df1['Signal'].nunique() < 2:
    st.write("Error: Only one class (Buy or Sell) found in the training data. Please adjust the RSI length or thresholds.")
else:
    # -----------------------------
    # Step 5: Train-Test Split
    # -----------------------------
    # Split the data into training and testing sets
    X = df1.RSI
    y = df1.Signal

    split = int(0.8 * len(X))

    X_train, X_test = X[:split].values.reshape(-1, 1), X[split:].values.reshape(-1, 1)
    y_train, y_test = y[:split], y[split:]

    # -----------------------------
    # Step 6: Fit Logistic Regression Model & Get Coefficient and Intercept
    # -----------------------------
    # Initialize and fit the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Get the coefficient and intercept
    a = model.coef_
    b = model.intercept_

    # -----------------------------
    # Step 7: Calculate Midpoint (50% point)
    # -----------------------------
    # Calculate midpoint where logistic function = 50%
    midpoint = -b / a

    # -----------------------------
    # Step 8: Plot the Logistic Regression Curve with Midpoint
    # -----------------------------
    # Define logistic function
    def logistic_func(X, a, b):
        return 1 / (1 + np.exp(-(a * X + b)))

    # Generate x-axis values for the smooth logistic curve
    X_new = np.linspace(min(X), max(X), 100)

    # Calculate y-values for the logistic function
    y_new = logistic_func(X_new, a, b).flatten()

    # -----------------------------
    # Step 9: Plot RSI and Logistic Regression
    # -----------------------------
    # Plot the RSI chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df['RSI'], label='RSI', color='blue')
    ax.axhline(lower_threshold, color='green', linestyle='--', label=f'Buy Threshold ({lower_threshold})')
    ax.axhline(upper_threshold, color='red', linestyle='--', label=f'Sell Threshold ({upper_threshold})')
    ax.set_title(f'RSI for {selected_stock}')
    ax.set_xlabel('Date')
    ax.set_ylabel('RSI')
    ax.legend()

    # Show RSI plot
    st.pyplot(fig)

    # Plot Logistic Regression Curve
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the data points and the logistic regression curve
    ax.scatter(X, y, label='Data Points')
    ax.plot(X_new, y_new, color='red', label='Logistic Curve')

    # Add vertical line at midpoint
    ax.axvline(midpoint[0][0], color='green', linestyle='--', label=f'Midpoint RSI = {midpoint[0][0]:.2f}')

    # Add horizontal line at y=0.5 (50% probability level)
    ax.axhline(0.5, color='blue', linestyle='--', label='50% Probability')

    # Customize the plot for better visualization
    ax.set_title('Logistic Regression Curve with Midpoint')
    ax.set_xlabel('RSI')
    ax.set_ylabel('Probability (BUY/SELL)')
    ax.legend()
    ax.grid(True)

    # Show logistic regression plot
    st.pyplot(fig)

    # -----------------------------
    # Step 10: Compare Predicted vs Actual
    # -----------------------------
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Create a comparison DataFrame
    comparison = pd.DataFrame({
        'Predicted': y_pred,
        'Actual': y_test
    })

    st.subheader("Predicted vs Actual")
    st.write(comparison)

    # -----------------------------
    # Step 11: Check Model Accuracy
    # -----------------------------
    # Calculate accuracy as percentage
    accuracy = accuracy_score(y_test, y_pred) * 100

    st.subheader("Model Accuracy")
    st.write(f"Accuracy: {accuracy:.2f}%")

    # -----------------------------
    # Step 12: Predict Example
    # -----------------------------
    # Predict the signal for an RSI value of 55
    test = model.predict([[55]])
    st.subheader(f'If we are given a Predicted signal for RSI = 55: {test[0]}')

    if test[0] == 1:
        st.write("Action: The model suggests a **BUY** signal. Consider purchasing the stock.")
    else:
        st.write("Action: The model suggests a **SELL** signal. Consider selling the stock.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("<br><br><hr><center><i>Created by Dr. Alvin Ang</i></center>", unsafe_allow_html=True)

