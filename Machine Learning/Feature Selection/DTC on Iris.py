import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Source
from sklearn import tree

# Load Dataset
@st.cache_data
def load_data():
    url = "https://www.alvinang.sg/s/iris_dataset.csv"
    iris_df = pd.read_csv(url)
    return iris_df

# Train Decision Tree Classifier
@st.cache_resource
def train_classifier(X, y):
    dtc = DecisionTreeClassifier(criterion="entropy")
    dtc.fit(X, y)
    return dtc

# Main Streamlit App
def main():
    # Title of the App
    st.title('Iris Dataset Decision Tree Classifier')

    # Description of the Objective
    st.write("""
    This app performs classification using a Decision Tree on the Iris dataset. 
    It ranks and visualizes feature importance and displays the trained decision tree.
    """)

    st.write("---")  # Separator line

    # Step 1: Load Dataset and Show Sample Data
    iris_df = load_data()
    st.subheader('Iris Dataset Preview')
    st.dataframe(iris_df.head())
    st.write("---")  # Single separator line after dataset preview

    # Define target and features
    y = iris_df["species"]
    X = iris_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]

    # Step 2: Train Decision Tree Classifier
    dtc = train_classifier(X, y)
    st.write("---")  # Separator line

    # Step 3: Rank the Feature Importance Scores and Visualize
    feature_importances_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": dtc.feature_importances_
    })

    feature_importances_df = feature_importances_df.sort_values(by="Importance", ascending=False)
    st.subheader('Feature Importance Scores')
    st.dataframe(feature_importances_df)

    st.write("""
    - The feature importance scores indicate how much each feature contributes to the decision-making process in the classifier.
    - Higher importance scores indicate that a feature has a stronger effect on the model's predictions.
    """)

    # Plotting the feature importance
    st.subheader('Feature Importance Visualization')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importances_df, palette="Set2", ax=ax)

    ax.set_xlabel("Feature Importance Score")
    ax.set_ylabel("Features")
    ax.set_title("Visualizing Important Features using Decision Tree Classifier")

    # Add the 0.95 threshold line
    ax.axvline(x=0.95, color='r', linestyle='dotted')
    ax.annotate('0.95', xy=(0.95, 3.3), xycoords='data', color='r')

    st.pyplot(fig)

    st.write("""
    - This bar plot shows the importance of each feature in predicting the target variable. The height of the bars represents the 
      strength of the relationship between the feature and the target.
    - The **0.95 threshold line** shows which features have a significant impact on predicting the target variable.
      Features that cross the 0.95 line are considered **significant** in prediction, while features below this threshold are less influential.
    """)

    st.write("---")  # Separator line

    # Step 4: Visualize the Decision Tree
    st.subheader('Visualizing the Decision Tree')

    graph = Source(tree.export_graphviz(dtc, out_file=None,
                                        feature_names=X.columns,
                                        class_names=pd.unique(iris_df["species"]),
                                        filled=True))

    graph.render("decision_tree", format="png", cleanup=False)
    
    # Display the decision tree image
    st.image("decision_tree.png", caption="Trained Decision Tree")

    st.write("""
    - This is the visual representation of the decision tree classifier. 
    - The tree splits the data based on feature values at each node, with the goal of classifying the input into one of the target species.
    - Each node represents a decision based on a feature value, and the branches represent the possible outcomes.
    - The leaf nodes show the predicted class, and the colors represent the majority class in each group.
    """)

    st.write("---")  # Separator line
    st.write("THE END")
    st.write("© 2025 Dr. Alvin Ang")

# Run the app
if __name__ == "__main__":
    main()

