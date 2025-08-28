import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score  # Import accuracy_score
from graphviz import Source
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

#--------------------------------------------------------------------
# Step 1: Load Dataset for Iris
#--------------------------------------------------------------------

@st.cache_data
def load_data():
    dataset_url = "https://www.alvinang.sg/s/iris_dataset.csv"
    iris_df = pd.read_csv(dataset_url)
    return iris_df

#--------------------------------------------------------------------
# Train Decision Tree Classifier for Feature Importance
#--------------------------------------------------------------------
@st.cache_resource
def train_classifier(X, y):
    dtc = DecisionTreeClassifier(criterion="entropy")
    dtc.fit(X, y)
    return dtc

#--------------------------------------------------------------------
# Main Streamlit App
#--------------------------------------------------------------------
def main():
    # Title of the App with an image
    st.title('🌸 Iris Dataset Decision Tree Classifier 🌺')

    # Description of the Objective
    st.write("""
    This app performs classification using a Decision Tree on the Iris dataset. 
    It ranks and visualizes feature importance, displays the trained decision tree, and provides model predictions.
    """)

    st.write("---")  # Separator line

    # Step 1: Load Dataset and Show Sample Data
    iris_df = load_data()
    st.subheader('🌿 Iris Dataset Preview 🌿')
    st.dataframe(iris_df.head())
    st.write("---")  # Single separator line after dataset preview

    # Define target and features
    y = iris_df["species"]
    X = iris_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]

    # Label encoding for target variable (species)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # Encode target labels

    # Step 2: Train Decision Tree Classifier
    dtc = train_classifier(X, y_encoded)

    #--------------------------------------------------------------------
    # Step 5: Pair Plot
    #--------------------------------------------------------------------
    st.header("📊 Pair Plot for Selected Features")

    # Define selected features
    selected_features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    # Create pair plot
    fig = sns.pairplot(iris_df[selected_features + ['species']])
    st.pyplot(fig)

    st.markdown("""
    **Pair Plot Description:**
    - The pair plot shows the relationships between pairs of features (e.g., Sepal Length vs. Petal Length) and how they relate to the target variable (Species).
    - **Sepal Length vs. Petal Length** shows a potential positive correlation with **Species**, where a larger petal length could indicate certain species.
    - **Petal Width and Sepal Width** also show some interesting trends in terms of the relationship with the target variable.
    - The diagonal shows the distribution of each feature, which helps in understanding its variance across species.
    """)

    st.markdown("---")

    #--------------------------------------------------------------------
    # Step 6: Correlation Heatmap
    #--------------------------------------------------------------------
    st.header("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(iris_df[selected_features].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.markdown("""
    **Correlation Heatmap Description:**
    - The correlation heatmap shows the relationships between pairs of features.
    - **Petal Length and Petal Width** show a very strong positive correlation, meaning they increase together.
    - **Sepal Length and Petal Length** also have a positive correlation, showing they are linked in the data.
    - The heatmap provides insight into which features are more strongly related, helping us understand how these features are likely influencing the target variable.
    """)

    st.markdown("---")

    #--------------------------------------------------------------------
    # Step 7: Feature Importance using Decision Tree
    #--------------------------------------------------------------------
    st.header("🔍 Feature Importance Using Decision Tree")

    # Rank the Feature Importance Scores
    feature_importances_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": dtc.feature_importances_
    })

    feature_importances_df = feature_importances_df.sort_values(by="Importance", ascending=False)
    st.subheader('Feature Importance Scores')
    st.dataframe(feature_importances_df)

    # Feature Importance Visualization
    st.subheader('Feature Importance Visualization')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importances_df, palette="Set2", ax=ax)

    ax.set_xlabel("Feature Importance Score")
    ax.set_ylabel("Features")
    ax.set_title("Visualizing Important Features using Decision Tree Classifier")
    ax.axvline(x=0.95, color='r', linestyle='dotted')
    ax.annotate('0.95', xy=(0.95, 3.3), xycoords='data', color='r')
    st.pyplot(fig)

    st.write("""
    - This chart shows which features are the most important for classifying the species.
    - The **0.95 threshold line** helps determine the significance of the features, where features crossing this threshold have a strong influence on the prediction.
    """)

    st.write("---")  # Separator line

    # Step 4: Visualize the Decision Tree
    st.subheader('🌳 Visualizing the Decision Tree 🌳')

    graph = Source(tree.export_graphviz(dtc, out_file=None,
                                        feature_names=X.columns,
                                        class_names=le.classes_,
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

    #--------------------------------------------------------------------
    # Sidebar - User Inputs for Prediction and Model Performance
    #--------------------------------------------------------------------
    st.sidebar.header("🎛️ Input Features for Prediction")

    sepal_length = st.sidebar.slider("📊 Sepal Length", 0.0, 8.0, 5.0, step=0.1)
    sepal_width = st.sidebar.slider("📈 Sepal Width", 0.0, 5.0, 3.0, step=0.1)
    petal_length = st.sidebar.slider("🔊 Petal Length", 0.0, 8.0, 4.0, step=0.1)
    petal_width = st.sidebar.slider("🎵 Petal Width", 0.0, 3.0, 1.0, step=0.1)

    # Make prediction for the user inputs
    user_input = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=X.columns)
    predicted_class = dtc.predict(user_input)
    
    # Inverse transform the predicted class index to species name
    predicted_class_name = le.inverse_transform(predicted_class)

    # Model Performance Metrics (updated)
    y_pred = dtc.predict(X)

    # Calculate accuracy
    accuracy = accuracy_score(y_encoded, y_pred)

    st.sidebar.markdown("---")
    st.sidebar.header("📊 Model Performance")

    # Display accuracy score in the sidebar
    st.sidebar.metric("📊 Accuracy", f"{accuracy * 100:.2f}%")

    # Provide a description of Accuracy
    st.sidebar.write("""
    **Accuracy**:
    - Accuracy measures the proportion of correctly predicted instances out of all instances.
    - **High accuracy** (typically above 80-90%) means the model is performing well.
    """)

    # Display prediction and model performance explanation
    st.sidebar.subheader("🔮 Prediction Result")
    st.sidebar.success(f"✅ Predicted Class: **{predicted_class_name[0]}**")

    st.markdown("---")

    #--------------------------------------------------------------------
    # Step 9: How to Use the App
    #--------------------------------------------------------------------

    st.markdown("### 📌 How to Use This App")
    st.write("""
    1. Adjust the **features** (Sepal Length, Sepal Width, Petal Length, Petal Width) using the sliders in the sidebar.
    2. Click on **Predict** to calculate the **predicted species**.
    3. View **data visualizations** (Pair plot, Correlation Heatmap, Feature Importance) and **model performance metrics**.
    """)

    # Add a horizontal line to indicate the end of the app
    st.markdown("---")
    st.markdown("**THE END**")
    st.markdown("© Dr. Alvin Ang")

# Run the app
if __name__ == "__main__":
    main()

