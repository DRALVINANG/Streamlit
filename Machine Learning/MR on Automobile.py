import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

# Generate correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg', 'price']].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

st.markdown("---")

#--------------------------------------------------------------------
# Step 5: Prediction Model
#--------------------------------------------------------------------
st.subheader("🔮 Predict Automobile Prices")

horsepower = st.slider("Select Horsepower:", min_value=int(df['horsepower'].min()), max_value=int(df['horsepower'].max()), value=150)
curb_weight = st.slider("Select Curb Weight (lbs):", min_value=int(df['curb-weight'].min()), max_value=int(df['curb-weight'].max()), value=2500)
engine_size = st.slider("Select Engine Size (cc):", min_value=int(df['engine-size'].min()), max_value=int(df['engine-size'].max()), value=120)
highway_mpg = st.slider("Select Highway MPG:", min_value=int(df['highway-mpg'].min()), max_value=int(df['highway-mpg'].max()), value=30)

if st.button("Predict and Visualize"):
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
    
    st.write(f"**Predicted Price:** ${predicted_price:.2f}")
    st.write(f"**R-squared Value:** {r2:.2f} ({r2_comment})")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f} ({mse_comment})")

st.markdown("---")

st.success("🎉 App Successfully Loaded! Adjust the sliders to start predicting.")
