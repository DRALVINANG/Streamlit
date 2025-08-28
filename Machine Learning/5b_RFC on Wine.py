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
# Step 1: Load Dataset for Wine
#--------------------------------------------------------------------
@st.cache_data
def load_data():
    dataset_url = "https://www.alvinang.sg/s/wine_sklearn_dataset.csv"
    wine_df = pd.read_csv(dataset_url)
    return wine_df

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
    st.title('🍷 Wine Dataset Random Forest Classifier 🍇')
    st.write("""
    This app performs classification using a Random Forest on the Wine dataset.
    It ranks and visualizes feature importance, displays model performance, and provides predictions.
    """)
    st.write("---")

    # Load Dataset
    wine_df = load_data()
    st.subheader('🍷 Wine Dataset Preview 🍷')
    st.dataframe(wine_df.head())

    #--------------------------------------------------------------------
    # Dataset Description
    #--------------------------------------------------------------------
    st.header("📖 Wine Dataset Description")
    st.markdown("""
    - **Chemical analysis of wines** from the same region in Italy, categorized into **three different wine classes** (0, 1, 2).
    - **178 wine samples**, each described by **13 chemical properties** that influence classification.
    
    **Feature Descriptions:**
    - **Alcohol**: Influences taste and body.
    - **Malic Acid**: Adds tartness and acidity.
    - **Ash**: Represents mineral content.
    - **Alcalinity of Ash**: Relates to balance and smoothness.
    - **Magnesium**: Affects structure and taste.
    - **Total Phenols**: Impacts flavor and astringency.
    - **Flavanoids**: Contribute to bitterness and antioxidants.
    - **Nonflavanoid Phenols**: Affect texture and stability.
    - **Proanthocyanins**: Influence tannins and color intensity.
    - **Color Intensity**: Determines depth of color.
    - **Hue**: Indicates wine shade and oxidation level.
    - **OD280/OD315**: Measures phenolic compound absorbance.
    - **Proline**: Linked to sweetness and wine quality.
    """)
    st.write("---")

    # Define target and features
    y = wine_df["target"]
    X = wine_df.drop(columns=["target"])

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    sliders = {}
    for col in X.columns:
        sliders[col] = st.sidebar.slider(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()), step=0.1)

    user_input = pd.DataFrame([sliders])
    predicted_class = rfc.predict(user_input)

    st.sidebar.subheader("🔮 Prediction Result")
    st.sidebar.success(f"✅ Predicted Class: **{predicted_class[0]}**")
    st.sidebar.markdown("---")

    #--------------------------------------------------------------------
    # Pair Plot
    #--------------------------------------------------------------------
    st.header("📊 Pair Plot for Selected Features")
    fig = sns.pairplot(wine_df, hue="target")
    st.pyplot(fig)
    st.write("---")

    #--------------------------------------------------------------------
    # Correlation Heatmap
    #--------------------------------------------------------------------
    st.header("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(wine_df.drop(columns=["target"]).corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
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
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importances_df, palette="Set2", ax=ax)
    ax.set_xlabel("Feature Importance Score")
    ax.set_ylabel("Features")
    ax.set_title("Visualizing Important Features using Random Forest Classifier")
    st.pyplot(fig)
    st.write("---")

    #--------------------------------------------------------------------
    # Visualizing a Tree from Random Forest
    #--------------------------------------------------------------------
    st.header("🌳 Visualizing a Tree from the Random Forest")
    estimators = rfc.estimators_
    tree_index = st.slider("Select a tree to visualize", 0, len(estimators) - 1, 0)
    dot_data = export_graphviz(estimators[tree_index], out_file=None,
                               feature_names=X.columns,
                               class_names=["Class 0", "Class 1", "Class 2"],
                               filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render("wine_random_forest_tree", format="png", cleanup=False)
    st.image("wine_random_forest_tree.png", caption=f"Tree {tree_index} from Random Forest")
    st.write("---")

    st.markdown("**THE END**")
    st.markdown("© Dr. Alvin Ang")

# Run the app
if __name__ == "__main__":
    main()

