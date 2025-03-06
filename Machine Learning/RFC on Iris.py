import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

#--------------------------------------------------------------------
# Step 1: Load Dataset for Iris
#--------------------------------------------------------------------
@st.cache_data
def load_data():
    dataset_url = "https://www.alvinang.sg/s/iris_dataset.csv"
    iris_df = pd.read_csv(dataset_url)
    return iris_df

#--------------------------------------------------------------------
# Train Random Forest Classifier
#--------------------------------------------------------------------
@st.cache_resource
def train_classifier(X_train, y_train):
    rfc = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42)
    rfc.fit(X_train, y_train)
    return rfc

#--------------------------------------------------------------------
# Main Streamlit App
#--------------------------------------------------------------------
def main():
    st.title('🌸 Iris Dataset Random Forest Classifier 🌺')
    st.write("""
    This app performs classification using a Random Forest on the Iris dataset.
    It ranks and visualizes feature importance, displays model performance, and provides predictions.
    """)
    st.write("---")

    # Load Dataset
    iris_df = load_data()
    st.subheader('🌿 Iris Dataset Preview 🌿')
    st.dataframe(iris_df.head())
    st.write("---")

    # Define target and features
    y = iris_df["species"]
    X = iris_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]

    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    rfc = train_classifier(X_train, y_train)

    #--------------------------------------------------------------------
    # Sidebar - Model Performance & Prediction Inputs
    #--------------------------------------------------------------------
    st.sidebar.header("📊 Model Performance")
    y_pred = rfc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.sidebar.metric("📊 Accuracy", f"{accuracy * 100:.2f}%")
    st.sidebar.write("Random Forest achieves high accuracy due to ensemble learning.")
    st.sidebar.markdown("---")
    
    st.sidebar.header("🎛️ Input Features for Prediction")
    sepal_length = st.sidebar.slider("📊 Sepal Length", 0.0, 8.0, 5.0, step=0.1)
    sepal_width = st.sidebar.slider("📈 Sepal Width", 0.0, 5.0, 3.0, step=0.1)
    petal_length = st.sidebar.slider("🔊 Petal Length", 0.0, 8.0, 4.0, step=0.1)
    petal_width = st.sidebar.slider("🎵 Petal Width", 0.0, 3.0, 1.0, step=0.1)

    user_input = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=X.columns)
    predicted_class = rfc.predict(user_input)
    predicted_class_name = le.inverse_transform(predicted_class)

    st.sidebar.subheader("🔮 Prediction Result")
    st.sidebar.success(f"✅ Predicted Class: **{predicted_class_name[0]}**")
    st.sidebar.markdown("---")

    #--------------------------------------------------------------------
    # Pair Plot
    #--------------------------------------------------------------------
    st.header("📊 Pair Plot for Selected Features")
    fig = sns.pairplot(iris_df, hue="species")
    st.pyplot(fig)
    st.markdown("""
    🔹 **Key Observations:**
    - Petal length and petal width show strong separability between species.
    - Sepal dimensions overlap more, making them less effective for classification.
    """)
    st.write("---")

    #--------------------------------------------------------------------
    # Correlation Heatmap
    #--------------------------------------------------------------------
    st.header("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(iris_df.drop(columns=["species"]).corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
    st.markdown("""
    🔹 **Key Observations:**
    - Petal length and petal width have a strong positive correlation, meaning they tend to increase together.
    - Sepal features show weaker correlations, making them less predictive for classification.
    """)
    st.write("---")

    #--------------------------------------------------------------------
    # Feature Importance
    #--------------------------------------------------------------------
    st.header("🔍 Feature Importance Using Random Forest")

    feature_importances_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rfc.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.subheader('Feature Importance Scores')
    st.dataframe(feature_importances_df)

    st.subheader('Feature Importance Visualization')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importances_df, palette="Set2", ax=ax)
    ax.set_xlabel("Feature Importance Score")
    ax.set_ylabel("Features")
    ax.set_title("Visualizing Important Features using Random Forest Classifier")
    st.pyplot(fig)
    st.markdown("""
    🔹 **Key Observations:**
    - Petal length and petal width contribute most to classification.
    - Sepal features are less influential, aligning with the patterns observed in previous charts.
    """)
    st.write("---")

    #--------------------------------------------------------------------
    # Visualizing One Decision Tree in Random Forest
    #--------------------------------------------------------------------
    st.header("🌳 Visualizing a Tree from the Random Forest")
    estimators = rfc.estimators_
    tree_index = st.slider("Select a tree to visualize", 0, len(estimators) - 1, 0)

    dot_data = export_graphviz(estimators[tree_index], out_file=None,
                               feature_names=X.columns,
                               class_names=le.classes_,
                               filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render("random_forest_tree", format="png", cleanup=False)
    st.image("random_forest_tree.png", caption=f"Tree {tree_index} from Random Forest")
    st.markdown("""
    🔹 **Key Observations:**
    - This tree is a single component of the Random Forest model.
    - Decision rules are based on feature values.
    - Multiple trees together improve classification performance by reducing overfitting.
    """)
    st.write("---")

    st.markdown("**THE END**")
    st.markdown("© Dr. Alvin Ang")

# Run the app
if __name__ == "__main__":
    main()
