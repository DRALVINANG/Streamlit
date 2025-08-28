import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm

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

# Add a horizontal line to segment the sidebar
st.sidebar.markdown("---")

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
# Step 4: OLS Model for Feature Importance
#--------------------------------------------------------------------

# Add constant to the features for OLS regression
X_ols = sm.add_constant(X)

# Fit OLS Model
ols_model = sm.OLS(y, X_ols).fit()

# Extract p-values from the OLS model
p_values = ols_model.pvalues[1:]  # exclude constant
p_values_sorted = p_values.sort_values(ascending=True)

#--------------------------------------------------------------------
# Step 5: Streamlit Page Layout
#--------------------------------------------------------------------

st.title("📊 Advertising Budget vs Sales - Multiple Linear Regression")

st.write("""
**Objective:** This app predicts sales based on TV, Radio, and Newspaper advertising budgets using Multiple Linear Regression.
""")

# Add a horizontal line to segment the objective and dataset preview
st.markdown("---")

st.markdown("### 📂 Dataset Preview")
st.dataframe(advert.head())

st.markdown("[📥 Download the Dataset](https://www.alvinang.sg/s/Advertising.csv)")

st.markdown("---")

#--------------------------------------------------------------------
# Step 6: Visualizations and Prediction Results
#--------------------------------------------------------------------

# Place Prediction Results in Sidebar
st.sidebar.subheader("🔮 Prediction Results and Model Performance")
st.sidebar.metric(label="📊 Predicted Sales (Units)", value=f"{predicted_sales:.2f}")
st.sidebar.metric("📉 R-squared", f"{r2_value:.2f}", help=f"Model Fit: {r2_feedback}")
st.sidebar.metric("📊 Mean Squared Error (MSE)", f"{mse_value:.2f}", help=f"Error Level: {mse_feedback}")

#--------------------------------------------------------------------
# Pair Plot with Description
#--------------------------------------------------------------------

st.subheader("📈 Data Visualizations")

# Pair Plot
st.markdown("#### 🔗 Pair Plot")
fig = sns.pairplot(advert)
st.pyplot(fig.fig)

st.markdown("""
**Pair Plot Description:**
The pair plot shows the relationships between each pair of features. Notably:
- **TV and Sales** show a strong positive correlation, meaning that higher TV advertising budgets tend to increase sales.
- **Radio and Sales** also show a positive correlation, though not as strong as with TV.
- **Newspaper and Sales** show a weaker positive relationship, indicating that newspaper advertising does not have as significant an impact on sales.
- The diagonal histograms represent the distribution of each feature.
""")

# Add a horizontal line to separate pair plot and correlation heatmap
st.markdown("---")

#--------------------------------------------------------------------
# Correlation Heatmap with Description
#--------------------------------------------------------------------

st.markdown("#### 🔥 Correlation Heatmap")
advert_clean = advert.drop(columns=['Unnamed: 0'])  # Remove the unnecessary column
fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(advert_clean.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
st.pyplot(fig)

st.markdown("""
**Correlation Heatmap Description:**
The heatmap illustrates the pairwise correlations between the features in the dataset:
- **TV and Sales** have a strong positive correlation of **0.90**, meaning that TV advertising has a high impact on sales.
- **Radio and Sales** have a positive correlation of **0.35**, showing a moderate relationship.
- **Newspaper and Sales** have a lower positive correlation of **0.20**, suggesting that the newspaper budget has a smaller effect on sales.
""")

st.markdown("---")

# Regression Plot
st.markdown("### 📊 Regression Plot (Actual vs Predicted Sales)")
fig, ax = plt.subplots(figsize=(5, 5))
sns.scatterplot(x=y_test, y=sales_pred_test)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", label="Ideal Prediction")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.legend()
st.pyplot(fig)

# Brief description of the regression plot
st.markdown("""
### 📊 Regression Plot (Actual vs Predicted Sales)
This plot compares the **actual sales** (on the x-axis) with the **predicted sales** (on the y-axis) from our model. Here’s what we can infer:
- **Points along the red line** represent perfect predictions, where the predicted sales are exactly equal to the actual sales.
- **Clusters of points** that are far from the red line indicate areas where the model has made significant errors in its predictions.
- The **closer the points are to the red line**, the better the model is at predicting sales.
""")

st.markdown("---")

#--------------------------------------------------------------------
# Step 7: Feature Importance Visualization
#--------------------------------------------------------------------

st.subheader("🔍 Feature Importance Analysis")

# Create a bar plot of 1 - P-value for feature importance
fig, ax = plt.subplots(figsize=(5, 5))
sns.barplot(x=1 - p_values_sorted, y=p_values_sorted.index, palette='Set2', ax=ax)

# Add a red dashed line for the 0.95 threshold
plt.axvline(x=0.95, color='r', linestyle='dotted')

# Annotate the threshold
plt.annotate('0.95', xy=(0.95, 2.5), xycoords='data', color='r')

# Add labels and title
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Visualizing Important Features using Multiple Regression')
st.pyplot(fig)

# Brief description of the feature importance plot
st.markdown("""
### 📊 Feature Importance Plot
This chart visualizes the importance of each feature in predicting the target variable (Sales) using Ordinary Least Squares (OLS) regression. The bars represent the feature importance scores, derived from the p-values. A **larger score** indicates that the feature has a **stronger and more statistically significant relationship** with the target variable.

The **red dashed line at 0.95** represents the typical significance threshold. Features with **scores above 0.95** are considered **significant**, indicating they have a strong relationship with the target variable, and thus are more important in predicting sales. Features with scores **below 0.95** have a weaker influence on sales and may not be as important for the model's prediction.
""")

st.markdown("---")

#--------------------------------------------------------------------
# Step 8: How to Use the App
#--------------------------------------------------------------------

st.markdown("### 📌 How to Use This App")
st.write("""
1. Adjust the **TV, Radio, and Newspaper budgets** using the sliders on the left.
2. The app will predict the **sales** and display performance metrics:
   - **R² Score**: Measures model fit (closer to 1 is better).
   - **MSE**: Measures prediction error (lower is better).
3. View **data visualizations** and model insights.
""")

# Add a horizontal line to indicate the end of the app
st.markdown("---")
st.markdown("**THE END**")
st.markdown("© Dr. Alvin Ang")

