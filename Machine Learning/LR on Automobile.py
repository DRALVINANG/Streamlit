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
df['highway-mpg'] = pd.to_numeric(df['highway-mpg'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df.dropna(inplace=True)  # Remove missing values

# Define X and Y
X = df[['highway-mpg']]
y = df['price']

# Initialize the Linear Regression model
lm = LinearRegression()
lm.fit(X, y)

# Predict values and calculate R² and MSE
r2 = lm.score(X, y)
mse = mean_squared_error(y, lm.predict(X))

#--------------------------------------------------------------------
# Streamlit App
#--------------------------------------------------------------------
st.title("🚗 Automobile Price Prediction Based on Highway-MPG")

st.markdown("""
### Objective:
This app demonstrates how to use **Linear Regression** to predict the price of a car based on its **highway-mpg** (fuel efficiency on highways). 
It includes visualizations and performance metrics to understand the model's behavior.

**Created by:** Dr. Alvin Ang
""")

st.markdown("---")

#--------------------------------------------------------------------
# Step 2: Display Dataset
#--------------------------------------------------------------------
st.subheader("📊 Dataset Preview")
st.write(df.head())
st.markdown("[📥 Download Dataset](https://www.alvinang.sg/s/automobileEDA.csv)")

st.markdown("---")

#--------------------------------------------------------------------
# Step 3: Data Visualization
#--------------------------------------------------------------------
st.subheader("📈 Visualize Relationships")
st.markdown("""
- **Pair Plot**: Shows the relationship between highway-mpg and price.
- **Correlation Heatmap**: Displays the correlation coefficient between highway-mpg and price.
""")

# Generate pair plot
fig = sns.pairplot(df[['highway-mpg', 'price']])
st.pyplot(fig)

# Generate correlation heatmap
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df[['highway-mpg', 'price']].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

st.markdown("---")

#--------------------------------------------------------------------
# Step 4: Prediction Model
#--------------------------------------------------------------------
st.subheader("🔮 Predict Car Price Based on Highway-MPG")
highway_mpg = st.slider("Select Highway MPG:", min_value=int(X.min()), max_value=int(X.max()), value=30)

if st.button("Predict and Visualize"):
    predicted_price = lm.predict([[highway_mpg]])[0]
    
    # Regression plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.regplot(x='highway-mpg', y='price', data=df, line_kws={'color': 'red'}, ax=ax)
    ax.scatter([highway_mpg], [predicted_price], color='orange', label=f'Predicted Price: ${predicted_price:.2f}', s=100)
    ax.set_xlabel('Highway-MPG')
    ax.set_ylabel('Price')
    ax.set_title('Regression Plot: Highway-MPG vs Price')
    ax.legend()
    ax.grid()
    st.pyplot(fig)
    
    # Model performance metrics
    mse_comment = "Good Fit" if mse <= 10 else "Poor Fit"
    r2_comment = "Excellent Fit" if r2 > 0.9 else "Acceptable Fit" if r2 > 0.7 else "Poor Fit"
    
    st.write(f"**Predicted Price:** ${predicted_price:.2f}")
    st.write(f"**R-squared Value:** {r2:.2f} ({r2_comment})")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f} ({mse_comment})")

st.markdown("---")

st.markdown("""
### How to Use This App:
1. Adjust the **highway-mpg** slider to predict the car price.
2. The app will display:
   - The **predicted price** of the car.
   - **R² Score (R-squared)**: Indicates how well the model fits the data.
   - **Mean Squared Error (MSE)**: Indicates the average squared difference between actual and predicted values.

**Performance Guidelines:**
- **R² Score:**
  - > 0.9: Excellent Fit
  - 0.7 - 0.9: Acceptable Fit
  - ≤ 0.7: Poor Fit
- **MSE:**
  - ≤ 10: Good Fit
  - > 100: Poor Fit
""")

st.markdown("---")

#--------------------------------------------------------------------
# End of App
#--------------------------------------------------------------------
st.success("🎉 App Successfully Loaded! Adjust the slider to start predicting.")
