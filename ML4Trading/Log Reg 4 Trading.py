import yfinance as yf
import seaborn as sns
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# -----------------------------
# Streamlit App Interface
# -----------------------------

# Title of the app
st.title('Stock Analysis with Logistic Regression and Backtesting')

# Introduction to guide the user
st.write("""
    **How to Use this App:**
    
    1. **Input a Stock Ticker**: Enter a stock ticker (e.g., AAPL for Apple) in the input box.
    2. **Adjust Time Period and Features**: The time period for all indicators is fixed at **14 days**. You can still select the features to include in the model.
    3. **Model Accuracy**: The app will display the model accuracy as a percentage.
    4. **Visualizations**: The app shows the stock's close price, RSI, logistic regression curve, confusion matrix, and backtest results.
    5. **Backtest**: You can view backtest performance (in terms of strategy returns) over a specified period.
    
    **Data Range Used**:
    The model is trained on historical data from **2022-01-01** to **2022-12-31**. The backtesting is done on data from **2023-01-01** to **2023-12-31**.
""")

# -----------------------------
# Step 1: Stock Ticker Input
# -----------------------------
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "D05.SI")
start_date = '2022-01-01'
end_date = '2022-12-31'

# Fixed time period for indicators
timeperiod = 14

# Checkbox to select which features to use
use_volatility = st.sidebar.checkbox("Use Volatility", value=True)
use_rsi = st.sidebar.checkbox("Use RSI", value=True)
use_adx = st.sidebar.checkbox("Use ADX", value=True)
use_corr = st.sidebar.checkbox("Use Correlation", value=True)

# -----------------------------
# Step 2: Download Data
# -----------------------------
# Download stock data
data = yf.download(ticker, start=start_date, end=end_date)
data.columns = data.columns.droplevel(level=1)

# -----------------------------
# Step 3: Candlestick Plot for the Training Data
# -----------------------------
st.subheader(f"Candlestick Plot for {ticker} (Training Data 2022)")
fig = go.Figure(data=[go.Candlestick(x=data.index,
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'],
                                     name="Candlestick")])

fig.update_layout(title=f'{ticker} Candlestick Chart (2022)',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  template='plotly_dark')
st.plotly_chart(fig)

# -----------------------------
# Step 4: Calculate Features
# -----------------------------
def get_target_features(data, timeperiod):
    features = []

    # Define Features (X)
    if use_volatility:
        data['PCT_CHANGE'] = data['Close'].pct_change()
        data['VOLATILITY'] = data['PCT_CHANGE'].rolling(timeperiod).std() * 100
        features.append('VOLATILITY')

    if use_rsi:
        data['RSI'] = ta.rsi(data['Close'], timeperiod=timeperiod)
        features.append('RSI')

    if use_adx:
        data['ADX'] = ta.adx(data['High'], data['Low'], data['Close'], length=timeperiod)['ADX_14']
        features.append('ADX')

    if use_corr:
        data['SMA'] = ta.sma(data['Close'], timeperiod=timeperiod)
        data['CORR'] = data['Close'].rolling(window=timeperiod).corr(data['SMA'])
        features.append('CORR')

    # Define Target (y)
    data['Returns_4_Tmrw'] = data['Close'].pct_change().shift(-1)
    data['Actual_Signal'] = np.where(data['Returns_4_Tmrw'] > 0, 1, 0)

    # Drop NaN rows
    data = data.dropna()

    return data['Actual_Signal'], data[features]

# Get the target and features
y, X = get_target_features(data, timeperiod)

# -----------------------------
# Step 5: Train-Test Split
# -----------------------------
split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# -----------------------------
# Step 6: Scale the Features
# -----------------------------
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# -----------------------------
# Step 7: Train the Logistic Regression Model
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# -----------------------------
# Step 8: Model Accuracy in Percentage
# -----------------------------
accuracy = metrics.accuracy_score(y_test, y_pred) * 100

# Display the accuracy as a percentage
st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}%")

# -----------------------------
# Step 9: Confusion Matrix
# -----------------------------
def get_metrics(y_test, predicted):
    confusion_matrix_data = metrics.confusion_matrix(y_test, predicted)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(confusion_matrix_data, fmt="d", cmap='Blues', cbar=False, annot=True, ax=ax)

    # Set axes labels and title
    ax.set_xlabel('Predicted Labels', fontsize=12)
    ax.set_ylabel('Actual Labels', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    ax.xaxis.set_ticklabels(['No Position', 'Long Position'])
    ax.yaxis.set_ticklabels(['No Position', 'Long Position'])

    # Display the plot
    st.pyplot(fig)

# Show confusion matrix
get_metrics(y_test, y_pred)

# -----------------------------
# Step 10: Backtesting the Model
# -----------------------------
# Get new data for backtesting (using 2023 data)
df_backtest = yf.download(ticker, start='2023-01-01', end='2023-12-31')
df_backtest.columns = df_backtest.columns.droplevel(level=1)

# Calculate Features for Backtest
features = []
if use_volatility:
    df_backtest['PCT_CHANGE'] = df_backtest['Close'].pct_change()
    df_backtest['VOLATILITY'] = df_backtest['PCT_CHANGE'].rolling(timeperiod).std() * 100
    features.append('VOLATILITY')

if use_rsi:
    df_backtest['RSI'] = ta.rsi(df_backtest['Close'], timeperiod=timeperiod)
    features.append('RSI')

if use_adx:
    df_backtest['ADX'] = ta.adx(df_backtest['High'], df_backtest['Low'], df_backtest['Close'], length=timeperiod)['ADX_14']
    features.append('ADX')

if use_corr:
    df_backtest['SMA'] = ta.sma(df_backtest['Close'], timeperiod=timeperiod)
    df_backtest['CORR'] = df_backtest['Close'].rolling(window=timeperiod).corr(df_backtest['SMA'])
    features.append('CORR')

df_backtest = df_backtest.dropna()

# Ensure the same features are selected in the backtest as during training
df_backtest_scaled = sc.transform(df_backtest[features].values)
df_backtest['predicted_signal_4_tmrw'] = model.predict(df_backtest_scaled)

# Calculate Strategy Returns
df_backtest['strategy_returns'] = df_backtest['predicted_signal_4_tmrw'].shift(1) * df_backtest['PCT_CHANGE']
df_backtest.dropna(inplace=True)

# -----------------------------
# Step 11: Manual Performance Stats (Pyfolio Alternative)
# -----------------------------
# Calculate cumulative strategy returns
df_backtest['cumulative_returns'] = (1 + df_backtest['strategy_returns']).cumprod() - 1

# Display performance stats (manually calculated)
st.subheader("Backtest Performance Stats")
st.write(f"Cumulative Returns: {df_backtest['cumulative_returns'].iloc[-1]:.2f}%")

# Display performance stats as a dataframe
st.subheader("Performance Stats (Dataframe)")
df_backtest['Date'] = df_backtest.index  # Add the index as a column
st.write(df_backtest[['Date', 'cumulative_returns']].tail())

# Plot backtest strategy performance
fig, ax = plt.subplots(figsize=(10, 6))
df_backtest['cumulative_returns'].plot(ax=ax)
ax.set_title('Cumulative Strategy Returns')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Returns')
st.pyplot(fig)

# -----------------------------
# Footer
# -----------------------------
st.markdown("<br><br><hr><center><i>Created by Dr. Alvin Ang</i></center>", unsafe_allow_html=True)

