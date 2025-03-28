import yfinance as yf
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st

st.set_option('client.showErrorDetails', False)

# -------------------------------------------------------------------------------------
# Step 1: Define Available Tech Stock Options
# -------------------------------------------------------------------------------------
tech_stocks = {
    'Apple': 'AAPL',
    'Microsoft': 'MSFT',
    'Google (Alphabet)': 'GOOGL',
    'Amazon': 'AMZN',
    'Tesla': 'TSLA',
    'Meta (Facebook)': 'META',
    'NVIDIA': 'NVDA'
}

# -------------------------------------------------------------------------------------
# Step 2: Streamlit App Interface
# -------------------------------------------------------------------------------------

# Title of the app
st.title('Tech Stock Price Prediction & Analysis')

# Description of the app
st.write("""
    This app allows you to predict the **closing price** of a selected tech stock based on its **Open**, **High**, and **Low** prices 
    from the previous trading day. It also includes an **interactive analysis** with pair plots to visualize relationships between 
    features, and a prediction model based on **linear regression**.
    
    Choose a stock from the dropdown list, adjust the input sliders to simulate a day’s stock prices, and see the predicted closing 
    price for the next day.
""")

# Dropdown for selecting the stock
selected_stock = st.sidebar.selectbox("Select a Tech Stock", list(tech_stocks.keys()))
ticker = tech_stocks[selected_stock]

# Date range for the analysis
start_date = '2020-01-01'
end_date = pd.to_datetime('today').strftime('%Y-%m-%d')

# Download data for the selected stock
data1 = yf.download(ticker, start=start_date, end=end_date)

# Flatten the multi-level columns in the DataFrame
data1.columns = data1.columns.droplevel(level=1)

# Show the date range being analyzed
st.sidebar.write(f"Date Range: {start_date} to {end_date}")

# -------------------------------------------------------------------------------------
# Step 3: Feature Engineering (Shifted Data)
# -------------------------------------------------------------------------------------
# Shift the features (Open, High, Low) by 1 to predict today's Close price
X = data1[['Open', 'High', 'Low']].shift(+1)

# Drop rows with NaN values resulting from the shift
X = X.dropna()

# Target variable is the Close price
y = data1['Close']

# Combine X and y into a single DataFrame for model fitting
df1 = X.join(y)

# Ensure that both X and y are aligned before fitting the model
X = df1[['Open', 'High', 'Low']]
y = df1['Close']

# Split data into training and test sets (80% training, 20% testing)
split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# -------------------------------------------------------------------------------------
# Step 4: Build the Model and Make Predictions
# -------------------------------------------------------------------------------------

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# -------------------------------------------------------------------------------------
# Step 5: Plot Pairplot for Feature Analysis (Fix for empty plot)
# -------------------------------------------------------------------------------------

# Ensure there is no missing data in the columns used for the pairplot
data1_clean = data1[['Open', 'High', 'Low', 'Close']].dropna()

# Check if there is data available for pairplot
if not data1_clean.empty:
    st.subheader("Feature Relationships - Pairplot")
    
    # Create pairplot directly (ensure data is numeric)
    sns.pairplot(data1_clean)
    st.pyplot()  # Let Streamlit automatically handle figure display
else:
    st.write("No valid data available for the pairplot. Please check for missing values.")

# -------------------------------------------------------------------------------------
# Step 6: Streamlit Input Sliders and Predictions
# -------------------------------------------------------------------------------------

# Input sliders for user to input custom values for Open, High, Low
st.sidebar.header('Stock Prediction Input Parameters')

open_price = st.sidebar.slider('Today\'s Open Price', min_value=0, max_value=1000, value=200)
high_price = st.sidebar.slider('Today\'s High Price', min_value=0, max_value=1000, value=200)
low_price = st.sidebar.slider('Today\'s Low Price', min_value=0, max_value=1000, value=200)

# Step 7: Model Prediction with User Inputs
prediction = model.predict([[open_price, high_price, low_price]])

# Display the result
st.subheader(f"Predicted Close Price for Tomorrow: ${prediction[0]:.2f}")

# -------------------------------------------------------------------------------------
# Step 8: R2 Score and Actual vs Predicted Comparison
# -------------------------------------------------------------------------------------
r2 = r2_score(y_test, y_pred)
st.write(f"Model R² Score: {r2:.4f}")

# Display the comparison of predicted vs actual values for the test set
comparison = pd.DataFrame({
    'Predicted': y_pred.tolist(),
    'Actual': y_test.tolist()
})

st.subheader("Predicted vs Actual Values")
st.write(comparison)

# -------------------------------------------------------------------------------------
# Step 9: Add Custom Features for Visualization (Optional)
# -------------------------------------------------------------------------------------

# Plot the actual vs predicted stock prices
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_test.index, y_test, label='Actual')
ax.plot(y_test.index, y_pred, label='Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.set_title(f'Actual vs Predicted Close Prices for {selected_stock}')
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)

# -------------------------------------------------------------------------------------
# Step 10: Footer
# -------------------------------------------------------------------------------------
st.markdown("<br><br><hr><center><i>Created by Dr. Alvin Ang</i></center>", unsafe_allow_html=True)

