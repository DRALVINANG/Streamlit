import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#--------------------------------------------------------------------
# Step 1: Load Dataset
#--------------------------------------------------------------------

@st.cache_data
def load_data():
    url = 'https://www.alvinang.sg/s/Advertising.csv'
    return pd.read_csv(url)

advert = load_data()

# Check if dataset loaded properly
if advert.empty:
    st.error("❌ Error: Dataset failed to load. Please check the URL or your internet connection.")
    st.stop()

# Ensure correct column names
required_columns = ['TV', 'Radio', 'Newspaper', 'Sales']
if not all(col in advert.columns for col in required_columns):
    st.error("❌ Error: The dataset does not have the expected columns.")
    st.stop()

#--------------------------------------------------------------------
# Step 2: Sidebar - User Inputs
#--------------------------------------------------------------------

st.sidebar.header("🎛️ Input Advertising Budgets ($)")
tv_budget = st.sidebar.slider("📺 TV Budget", 0, 500, 100, 10)
radio_budget = st.sidebar.slider("📻 Radio Budget", 0, 500, 50, 10)
newspaper_budget = st.sidebar.slider("📰 Newspaper Budget", 0, 500, 20, 10)

#--------------------------------------------------------------------
# Step 3: Model Training & Prediction
#--------------------------------------------------------------------

X = advert[['TV', 'Radio', 'Newspaper']]
y = advert['Sales']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict Sales based on user input
predicted_sales = model.predict([[tv_budget, radio_budget, newspaper_budget]])[0]

# Predict on test set for evaluation
sales_pred_test = model.predict(X_test)

# Calculate R² and MSE
r2_value = r2_score(y_test, sales_pred_test)
mse_value = mean_squared_error(y_test, sales_pred_test)

# Performance Feedback
r2_feedback = "Excellent Fit" if r2_value > 0.9 else "Acceptable Fit" if r2_value >= 0.7 else "Poor Fit"
mse_feedback = "Good Fit" if mse_value <= 10 else "Moderate Fit" if mse_value <= 100 else "Poor Fit"

#--------------------------------------------------------------------
# Step 4: Streamlit Page Layout
#--------------------------------------------------------------------

st.title("📊 Advertising Budget vs Sales - Multiple Linear Regression")

st.write("""
**Objective:** This app predicts sales based on TV, Radio, and Newspaper advertising budgets using Multiple Linear Regression.
""")

st.markdown("### 📂 Dataset Preview")
st.dataframe(advert.head())

st.markdown("[📥 Download the Dataset](https://www.alvinang.sg/s/Advertising.csv)")

st.markdown("---")

#--------------------------------------------------------------------
# Step 5: Generate Visualizations
#--------------------------------------------------------------------

st.subheader("📈 Data Visualizations")

# Pair Plot Fix
st.markdown("#### 🔗 Pair Plot")
fig = sns.pairplot(advert)
st.pyplot(fig.fig)

# Correlation Heatmap
st.markdown("#### 🔥 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(advert.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
st.pyplot(fig)

st.markdown("---")

#--------------------------------------------------------------------
# Step 6: Display Prediction Results
#--------------------------------------------------------------------

st.subheader("🔮 Predicted Sales and Model Performance")

st.metric(label="📊 Predicted Sales (Units)", value=f"{predicted_sales:.2f}")

st.markdown("### 📌 Model Performance")
col1, col2 = st.columns(2)
col1.metric("📉 R-squared", f"{r2_value:.2f}", help=f"Model Fit: {r2_feedback}")
col2.metric("📊 Mean Squared Error (MSE)", f"{mse_value:.2f}", help=f"Error Level: {mse_feedback}")

# Regression Plot
st.markdown("### 📊 Regression Plot (Actual vs Predicted Sales)")
fig, ax = plt.subplots(figsize=(5, 5))
sns.scatterplot(x=y_test, y=sales_pred_test)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", label="Ideal Prediction")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.legend()
st.pyplot(fig)

st.markdown("---")

st.markdown("### 📌 How to Use This App")
st.write("""
1. Adjust the **TV, Radio, and Newspaper budgets** using the sliders on the left.
2. The app will predict the **sales** and display performance metrics:
   - **R² Score**: Measures model fit (closer to 1 is better).
   - **MSE**: Measures prediction error (lower is better).
3. View **data visualizations** and model insights.
""")

st.markdown("**Created by:** Dr. Alvin Ang")

