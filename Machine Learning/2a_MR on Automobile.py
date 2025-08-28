import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

#--------------------------------------------------------------------
# Step 1: Load Dataset
#--------------------------------------------------------------------
url = 'https://www.alvinang.sg/s/automobileEDA.csv'
df = pd.read_csv(url)

# Ensure correct column names and data types
df.rename(columns=lambda x: x.strip(), inplace=True)
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df['curb-weight'] = pd.to_numeric(df['curb-weight'], errors='coerce')
df['engine-size'] = pd.to_numeric(df['engine-size'], errors='coerce')
df['highway-mpg'] = pd.to_numeric(df['highway-mpg'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df.dropna(inplace=True)  # Remove missing values

# Define X and Y
X = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
y = df['price']

# Initialize the Linear Regression model
model = LinearRegression()
model.fit(X, y)

#--------------------------------------------------------------------
# Streamlit App
#--------------------------------------------------------------------
st.title("🚗 Automobile Price Prediction - Linear Regression")

st.markdown("""
### Objective:
This app demonstrates how to use **Linear Regression** to predict the price of a car based on its **horsepower, curb weight, engine size, and highway-mpg**.
It includes visualizations and performance metrics to understand the model's behavior.
""")

st.markdown("---")

#--------------------------------------------------------------------
# Step 2: Dataset Description
#--------------------------------------------------------------------
st.subheader("📖 About the Dataset")

st.markdown("""
The **Automobile Dataset** from the **UCI Machine Learning Repository** provides data for predicting the price of different types of cars based on their attributes.
The dataset contains **26 features** related to various characteristics of automobiles.

**Key Features Used in Prediction:**
- **Horsepower:** Engine power in horsepower (hp), indicating the performance of the vehicle.
- **Curb Weight:** Weight of the car without passengers or cargo (lbs).
- **Engine Size:** Volume of the engine in cubic centimeters (cc), affecting power output and efficiency.
- **Highway MPG:** Fuel efficiency of the car on highways (miles per gallon, mpg).
- **Price (Target Variable):** The price of the car in dollars, which we aim to predict.

For more details, visit:
- [UCI Automobile Dataset](https://archive.ics.uci.edu/dataset/10/automobile)
- [Original Dataset](https://github.com/DRALVINANG/Machine-Learning-with-Python-Training/blob/main/Linear%20Regression/Automobile.csv)
- [Cleaned Dataset](https://www.alvinang.sg/s/automobileEDA.csv)
""")

st.markdown("---")

#--------------------------------------------------------------------
# Step 3: Display Dataset
#--------------------------------------------------------------------
st.subheader("📊 Dataset Preview")
st.write(df.head())
st.markdown("[📥 Download Dataset](https://www.alvinang.sg/s/automobileEDA.csv)")

st.markdown("---")

#--------------------------------------------------------------------
# Step 4: Data Visualization
#--------------------------------------------------------------------
st.subheader("📈 Visualize Relationships")

# Generate pair plot
fig = sns.pairplot(df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg', 'price']])
st.pyplot(fig)

st.markdown("""
**Pair Plot Description:**
- **TV and Sales** show a strong positive correlation, meaning that higher TV advertising budgets tend to increase sales.
- **Radio and Sales** also show a positive correlation, though not as strong as with TV.
- **Newspaper and Sales** show a weaker positive relationship, indicating that newspaper advertising does not have as significant an impact on sales.
- The diagonal histograms represent the distribution of each feature.
""")

# Add a horizontal line to separate pair plot and correlation heatmap
st.markdown("---")

# Generate correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg', 'price']].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

st.markdown("""
**Correlation Heatmap Description:**
- **Horsepower and Price** have a strong positive correlation, meaning higher horsepower tends to increase the price.
- **Curb-Weight and Price** also show a moderate positive correlation.
- **Engine Size and Price** show a strong positive correlation.
- **Highway MPG and Price** show a weaker positive correlation.
""")

st.markdown("---")

#--------------------------------------------------------------------
# Step 5: Prediction Model
#--------------------------------------------------------------------

# Place sliders and prediction results in sidebar
st.sidebar.subheader("🔮 Prediction Results and Model Performance")
horsepower = st.sidebar.slider("Select Horsepower:", min_value=int(df['horsepower'].min()), max_value=int(df['horsepower'].max()), value=150)
curb_weight = st.sidebar.slider("Select Curb Weight (lbs):", min_value=int(df['curb-weight'].min()), max_value=int(df['curb-weight'].max()), value=2500)
engine_size = st.sidebar.slider("Select Engine Size (cc):", min_value=int(df['engine-size'].min()), max_value=int(df['engine-size'].max()), value=120)
highway_mpg = st.sidebar.slider("Select Highway MPG:", min_value=int(df['highway-mpg'].min()), max_value=int(df['highway-mpg'].max()), value=30)

# When button is clicked, show predicted price
if st.sidebar.button("Predict and Visualize"):
    predicted_price = model.predict([[horsepower, curb_weight, engine_size, highway_mpg]])[0]
    
    # Predict for the whole dataset for visualization
    y_pred = model.predict(X)
    
    # Create Actual vs Predicted Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(df['price'], color="r", label="Actual Price")
    sns.kdeplot(y_pred, color="b", label="Predicted Price")
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Density')
    plt.legend()
    st.pyplot(fig)
    
    # Create Residual Plot
    residuals = y - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.residplot(x=y_pred, y=residuals, lowess=True, color="g", ax=ax)
    ax.set_xlabel("Predicted Price")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot")
    st.pyplot(fig)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    mse_comment = "Good Fit" if mse <= 1e6 else "Poor Fit"
    r2_comment = "Excellent Fit" if r2 > 0.9 else "Acceptable Fit" if r2 > 0.7 else "Poor Fit"
    
    st.sidebar.write(f"**Predicted Price:** ${predicted_price:.2f}")
    st.sidebar.write(f"**R-squared Value:** {r2:.2f} ({r2_comment})")
    st.sidebar.write(f"**Mean Squared Error (MSE):** {mse:.2f} ({mse_comment})")

st.markdown("---")

#--------------------------------------------------------------------
# Step 6: Feature Importance using MR
#--------------------------------------------------------------------
st.subheader("🔍 Feature Importance using Multiple Regression")

# Add constant to the features for OLS regression
X_ols = sm.add_constant(X)

# Fit OLS Model
ols_model = sm.OLS(y, X_ols).fit()

# Extract p-values from the OLS model
p_values = ols_model.pvalues[1:]  # exclude constant
p_values_sorted = p_values.sort_values(ascending=True)

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

st.markdown("""
### 📊 Feature Importance Plot
This chart visualizes the importance of each feature in predicting the target variable (Price) using Ordinary Least Squares (OLS) regression. The bars represent the feature importance scores, derived from the p-values. A **larger score** indicates that the feature has a **stronger and more statistically significant relationship** with the target variable.

The **red dashed line at 0.95** represents the typical significance threshold. Features with **scores above 0.95** are considered **significant**, indicating they have a strong relationship with the target variable, and thus are more important in predicting price. Features with scores **below 0.95** have a weaker influence on price and may not be as important for the model's prediction.
""")

st.markdown("---")

#--------------------------------------------------------------------
# Step 7: How to Use the App
#--------------------------------------------------------------------

st.markdown("### 📌 How to Use This App")
st.write("""
1. Adjust the **horsepower, curb weight, engine size, and highway MPG** using the sliders on the left.
2. The app will predict the **price** and display performance metrics:
   - **R² Score**: Measures model fit (closer to 1 is better).
   - **MSE**: Measures prediction error (lower is better).
3. View **data visualizations** and model insights.
""")

# Add a horizontal line to indicate the end of the app
st.markdown("---")
st.markdown("**THE END**")
st.markdown("© Dr. Alvin Ang")

