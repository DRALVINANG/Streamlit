import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from statsmodels.api import OLS
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
@st.cache_data
def load_data():
    iris_df = pd.read_csv("https://www.alvinang.sg/s/iris_dataset.csv")
    return iris_df

# Fit OLS Model
@st.cache_resource
def fit_ols_model(X, y):
    model = OLS(y, X).fit()
    return model

# Main Streamlit App
def main():
    # Title of the App
    st.title('Iris Dataset Regression Analysis')

    # Description of the Objective
    st.write("""
    This app performs regression analysis on the Iris dataset to examine how different features 
    (such as sepal length, sepal width, petal length, and petal width) relate to the target variable.
    It also visualizes feature importance and provides insights on which features have the strongest 
    predictive power for the target variable.
    """)
    
    st.write("---")  # Separator line

    # Step 1: Load Dataset and Show Sample Data
    iris_df = load_data()
    st.subheader('Iris Dataset Preview')
    st.dataframe(iris_df.head())
    st.write("---")  # Single separator line after dataset preview

    # Label Encoding for Target Variable
    le = LabelEncoder()
    y = le.fit_transform(iris_df["species"])
    X = iris_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]

    # Step 2: Fit OLS Model
    model = fit_ols_model(X, y)
    st.write("---")  # Separator line

    # Step 3: Extract P-Values for Feature Importance
    p_values = pd.DataFrame((model.pvalues), columns=["P-value"]).sort_values(by="P-value", ascending=True)
    st.subheader('P-Values for Feature Importance')
    st.dataframe(p_values)

    st.write("""
    - **P-values** are used to determine the statistical significance of each feature in the regression model.
    - A lower p-value (typically less than 0.05) suggests that the feature has a strong relationship with the target variable.
    - Higher p-values suggest that the feature has a weaker or less significant relationship.
    """)

    st.write("---")  # Separator line

    # Step 4: Combine DataFrame with Encoded Target
    combined_df = pd.concat([iris_df, pd.DataFrame({"target": y})], axis=1)
    combined_df = combined_df.drop("species", axis=1)

    # Step 5: Visualize Feature Importance with Bar Plot
    st.subheader('Feature Importance Visualization')
    fig2, ax2 = plt.subplots(figsize=(8, 6))  # Create a new figure for the bar plot
    sns.barplot(x=1 - p_values['P-value'], y=p_values.index, palette='Set2', ax=ax2)

    ax2.axvline(x=0.95, color='r', linestyle='dotted')
    ax2.annotate('0.95', xy=(0.95, 3.3), xycoords='data', color='r')

    ax2.set_xlabel('Feature Importance Score')
    ax2.set_ylabel('Features')
    ax2.set_title('Visualizing Important Features using Multiple Regression')

    st.pyplot(fig2)

    st.write("""
    - This bar plot shows the importance of each feature in predicting the target variable. The height of the bars represents the 
      strength of the relationship between the feature and the target.
    - The **0.95 threshold line** shows which features have a significant impact on predicting the target variable.
      Features that cross the 0.95 line are considered **significant** in prediction, while features below this threshold are less influential.
    """)

    st.write("---")  # Separator line

    # The End of the app
    st.write("THE END")
    st.write("© 2025 Dr. Alvin Ang")

# Run the app
if __name__ == "__main__":
    main()

