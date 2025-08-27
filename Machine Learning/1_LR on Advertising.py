import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

#--------------------------------------------------------------------
# Step 1: Load Dataset
#--------------------------------------------------------------------
data = {
    "TV Advertising Budget ($)": [100, 150, 200, 250, 300],
    "Sales (units)": [20, 25, 30, 35, 40]
}
advert = pd.DataFrame(data)

#--------------------------------------------------------------------
# Step 2: Train Linear Regression Model
#--------------------------------------------------------------------
X = advert[["TV Advertising Budget ($)"]]
y = advert["Sales (units)"]

model = LinearRegression()
model.fit(X, y)

m = model.coef_[0]  # Slope
c = model.intercept_  # Intercept

#--------------------------------------------------------------------
# Step 3: Define Functions for Visualizations
#--------------------------------------------------------------------
def generate_visualizations():
    st.subheader("Pair Plot: Scatter Plot of TV Advertising vs Sales")
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.scatterplot(data=advert, x="TV Advertising Budget ($)", y="Sales (units)", ax=ax)
    plt.title("TV Advertising vs Sales")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(advert.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    plt.title("Correlation Heatmap")
    st.pyplot(fig)

#--------------------------------------------------------------------
# Step 4: Define Prediction Function
#--------------------------------------------------------------------
def predict_and_visualize_with_feedback(tv_budget):
    predicted_sales = model.predict([[tv_budget]])[0]
    r2_value = model.score(X, y)
    
    # Regression plot
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.scatterplot(data=advert, x="TV Advertising Budget ($)", y="Sales (units)", ax=ax)
    x_range = np.linspace(0, 500, 100)
    y_range = model.predict(x_range.reshape(-1, 1))
    ax.plot(x_range, y_range, color="red", label=f"Regression Line (y = {m:.2f}x + {c:.2f})")
    plt.title("Regression Plot")
    plt.legend()
    st.pyplot(fig)

    # Display results
    st.write(f"### Predicted Sales: {predicted_sales:.2f} units")
    st.write(f"*R² Value:* {r2_value:.2f}")
    st.write(f"### Regression Equation: y = {m:.2f}x + {c:.2f}")

#--------------------------------------------------------------------
# Step 5: Streamlit UI Setup
#--------------------------------------------------------------------
st.title("TV Advertising vs Sales - Linear Regression")

st.markdown("""
*Objective:* This app demonstrates how to use Linear Regression to predict sales based on TV advertising budgets.

*Created by:* Dr. Alvin Ang
""")

#--------------------------------------------------------------------
# Step 6: Dataset Preview
#--------------------------------------------------------------------
st.subheader("Dataset Preview")
st.dataframe(advert)

st.markdown("[Download the Dataset](https://www.alvinang.sg/s/Advertising.csv)")

#--------------------------------------------------------------------
# Step 7: Generate Visualizations
#--------------------------------------------------------------------
if st.button("Generate Visualizations"):
    generate_visualizations()

#--------------------------------------------------------------------
# Step 8: User Input for Prediction
#--------------------------------------------------------------------
st.subheader("Make Predictions")
tv_budget = st.slider("Select TV Advertising Budget ($)", min_value=0, max_value=500, step=10, value=100)

#--------------------------------------------------------------------
# Step 9: Predict and Visualize
#--------------------------------------------------------------------
if st.button("Predict and Visualize"):
    predict_and_visualize_with_feedback(tv_budget)

#--------------------------------------------------------------------
# Step 10: Conclusion
#--------------------------------------------------------------------
st.subheader("Conclusion")
st.markdown("""
This analysis demonstrates how TV advertising budgets influence product sales using a simple linear regression model.
Key takeaways:
- A higher TV advertising budget generally leads to increased sales.
- The model fits the data well, with an automatically computed regression equation.
- The regression equation (y = mx + c) provides a more accurate prediction of sales.
- Future enhancements could involve incorporating more advertising channels to refine predictions.
""")
