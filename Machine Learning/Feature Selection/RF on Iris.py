import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import graphviz
from sklearn.tree import export_graphviz

# Load Dataset
@st.cache_data
def load_data():
    url = "https://www.alvinang.sg/s/iris_dataset.csv"
    iris_df = pd.read_csv(url)
    return iris_df

# Train Random Forest Classifier
@st.cache_resource
def train_classifier(X, y):
    rfc = RandomForestClassifier(n_estimators=3, criterion="entropy", random_state=42)
    rfc.fit(X, y)
    return rfc

# Main Streamlit App
def main():
    # Title of the App
    st.title('Iris Dataset Random Forest Classifier')

    # Description of the Objective
    st.write("""
    This app performs classification using a Random Forest Classifier on the Iris dataset. 
    It ranks and visualizes feature importance and displays the decision trees from the trained Random Forest model.
    """)

    st.write("---")  # Separator line

    # Step 1: Load Dataset and Show Sample Data
    iris_df = load_data()
    st.subheader('Iris Dataset Preview')
    st.dataframe(iris_df.head())
    st.write("""
    - The dataset consists of 150 samples of iris flowers, with four features: 
      sepal length, sepal width, petal length, and petal width.
    - The target variable is the species of the flower, with three possible classes: 
      Setosa, Versicolor, and Virginica.
    """)
    st.write("---")  # Separator line

    # Define target and features
    y = iris_df["species"]
    X = iris_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]

    # Step 2: Train Random Forest Classifier
    rfc = train_classifier(X, y)
    st.write("---")  # Separator line

    # Step 3: Rank the Feature Importance Scores and Visualize
    feature_importances_rfc = rfc.feature_importances_

    feature_importances_df_rfc = pd.DataFrame(
        data={"Feature": X.columns, "Importance": feature_importances_rfc}
    )

    feature_importances_df_rfc = feature_importances_df_rfc.sort_values(by="Importance", ascending=False)

    st.subheader('Feature Importance Scores')
    st.dataframe(feature_importances_df_rfc)

    st.write("""
    - **Feature Importance** is a technique used to understand which features have the most influence on the model.
    - Features with higher importance scores are more influential in making predictions.
    - These scores help in determining which features contribute the most to the Random Forest's decision-making process.
    """)

    # Add separator line between table and plot
    st.write("---")  # Separator line

    # Feature Importance Visualization
    st.subheader('Feature Importance Visualization')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importances_df_rfc, palette="Set2", ax=ax)

    ax.set_xlabel("Feature Importance Score")
    ax.set_ylabel("Features")
    ax.set_title("Visualizing Important Features using Random Forest Classifier")

    # Add a threshold line at 0.95 (explanation follows)
    ax.axvline(x=0.95, color='r', linestyle='dotted')
    ax.annotate('0.95', xy=(0.95, 3.3), xycoords='data', color='r')

    st.pyplot(fig)

    st.write("""
    - This bar plot visualizes the importance of each feature in predicting the target variable. The taller the bar, the more important the feature is.
    - The **red dashed line at 0.95** marks the threshold for highly important features. 
    - However, **none of the features cross this threshold**, which suggests that while these features are important, they are not dominant enough to reach this cutoff.
    - The absence of features crossing the 0.95 line does **not mean that all features are unimportant**. It simply means that none of them are "extremely dominant" in predicting the target variable. All features still play a role in classification.
    """)

    st.write("---")  # Separator line

    # Step 4: Visualize the Trees in the Random Forest
    st.subheader('Visualizing the Decision Trees in the Random Forest')

    for i, estimator in enumerate(rfc.estimators_):
        dot_data = export_graphviz(estimator, out_file=None,
                                   feature_names=X.columns,
                                   class_names=["Setosa", "Versicolor", "Virginica"],
                                   filled=True, rounded=True)
        graph = graphviz.Source(dot_data)
        st.subheader(f'Decision Tree {i + 1}')
        st.graphviz_chart(dot_data)

    st.write("""
    - Each decision tree in the Random Forest is built by making splits based on feature values.
    - The tree aims to classify the data into one of the target classes by making decisions at each node.
    - The color of each node represents the majority class of that particular split.
    - The leaf nodes show the final predicted class, with the color corresponding to the predicted class.
    """)

    st.write("---")  # Separator line
    st.write("THE END")
    st.write("© 2025 Dr. Alvin Ang")

# Run the app
if __name__ == "__main__":
    main()

