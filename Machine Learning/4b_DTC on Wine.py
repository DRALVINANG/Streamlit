import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Source
from sklearn import tree

# Load Dataset
@st.cache_data
def load_data():
    url = "https://www.alvinang.sg/s/wine_small.csv"
    wine_df = pd.read_csv(url)
    return wine_df

# Train Decision Tree Classifier
@st.cache_resource
def train_classifier(X, y):
    dtc = DecisionTreeClassifier(criterion="entropy")
    dtc.fit(X, y)
    return dtc

# Main Streamlit App
def main():
    # Title of the App
    st.title('Wine Dataset Decision Tree Classifier')

    # Description of the Objective
    st.write("""
    This app performs classification using a Decision Tree on the Wine dataset.
    It ranks and visualizes feature importance, displays the trained decision tree,
    and evaluates the model's performance.
    """)

    st.write("---")  # Separator line

    # Step 1: Load Dataset and Show Sample Data
    wine_df = load_data()
    st.subheader('Wine Dataset Preview')
    st.dataframe(wine_df.head())
    st.write("---")  # Separator line

    # Define target and features
    y = wine_df["target"]
    X = wine_df[["alcohol", "flavanoids", "color_intensity"]]

    # Step 2: Pair Plot
    st.header("📊 Pair Plot")
    st.write("""
    - The pair plot shows the relationships between pairs of features (e.g., Alcohol vs. Flavonoids) and how they relate to the target variable (Wine Class).
    - Each feature is compared to every other feature in a matrix of scatterplots.
    - The diagonal line plots the distribution of each feature.
    """)
    
    fig = sns.pairplot(wine_df[["alcohol", "flavanoids", "color_intensity", "target"]], hue="target")
    st.pyplot(fig)

    st.write("""
    **Observations from the Pair Plot**:
    - **Alcohol vs Flavonoids**: The scatter plot shows some separation between Wine Class 1 and Class 3. There is a weak positive correlation, but overlap between Wine Class 1 and Class 2 suggests that alcohol and flavanoids are not perfectly discriminating features.
    - **Alcohol vs Color Intensity**: There’s a weak positive correlation between alcohol and color intensity. Wine Class 2 and Class 3 seem more separated than Class 1, which indicates that while color intensity and alcohol may have some relationship, they are not strong discriminators by themselves.
    - **Flavonoids vs Color Intensity**: There’s little to no separation between the wine classes in this scatter plot. The overlap between all classes suggests that these features are not very useful for distinguishing between wine classes.
    """)

    st.write("---")  # Separator line

    # Step 3: Correlation Heatmap
    st.header("🔥 Correlation Heatmap")
    st.write("""
    - The correlation heatmap shows the relationships between pairs of features.
    - The colors represent the strength of the correlation: blue for positive, red for negative.
    - Strong correlations (near 1 or -1) indicate that the features are strongly related.
    """)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(wine_df[["alcohol", "flavanoids", "color_intensity"]].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.write("""
    **Observations from the Correlation Heatmap**:
    - **Alcohol vs Flavonoids**: There is a weak positive correlation (**0.24**) between alcohol and flavanoids, indicating a slight relationship between these two features.
    - **Alcohol vs Color Intensity**: The correlation (**0.55**) between alcohol and color intensity is moderate, showing a moderate positive relationship, where wines with higher alcohol content tend to have higher color intensity.
    - **Flavonoids vs Color Intensity**: The correlation (**-0.17**) is weak and negative, suggesting that as flavanoids increase, color intensity slightly decreases, but the relationship is not strong.
    """)

    st.write("---")  # Separator line

    # Step 4: Train Decision Tree Classifier
    dtc = train_classifier(X, y)

    # Step 5: Rank the Feature Importance Scores and Visualize
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
    **Observations from the Feature Importance Chart**:
    - All feature importance scores are below **0.95**, meaning none of the features have an extremely dominant impact on the classification decision.
    - The **Flavonoids** feature has the highest importance at **0.39**, followed by **Alcohol** at **0.33**, and **Color Intensity** at **0.28**.
    - While none of the features are "extremely significant" (i.e., they don’t surpass the **0.95** threshold), they all contribute to the classification. In real-world scenarios, even relatively modest importance scores can be valuable when combined.
    - The absence of any feature surpassing the **0.95** threshold doesn’t necessarily mean that the features are insignificant. It simply indicates that no single feature overwhelmingly determines the class of the wine, and a combination of features is necessary for accurate classification.
    """)

    st.write("---")  # Separator line

    # Step 6: Visualize the Decision Tree
    st.subheader('Visualizing the Decision Tree')

    # Convert class names to strings (fix for the TypeError)
    class_names = [str(class_name) for class_name in pd.unique(wine_df["target"])]
    
    # Export the decision tree visualization
    graph = Source(tree.export_graphviz(dtc, out_file=None,
                                        feature_names=X.columns,
                                        class_names=class_names,  # Ensure class names are strings
                                        filled=True))

    graph.render("decision_tree_wine", format="png", cleanup=False)
    
    # Display the decision tree image
    st.image("decision_tree_wine.png", caption="Trained Decision Tree")

    st.write("""
    - This is the visual representation of the decision tree classifier. 
    - The tree splits the data based on feature values at each node, with the goal of classifying the input into one of the target wine classes.
    - Each node represents a decision based on a feature value, and the branches represent the possible outcomes.
    - The leaf nodes show the predicted class, and the colors represent the majority class in each group.
    """)

    st.write("---")  # Separator line

    # Sidebar for interactive input and predictions
    with st.sidebar:
        # Step 7: Model Testing and Accuracy Score
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Train the classifier
        dtc.fit(X_train, y_train)

        # Predict on the test set
        y_pred = dtc.predict(X_test)

        # Calculate and display the accuracy score
        accuracy = accuracy_score(y_test, y_pred)
        st.subheader('Model Accuracy')
        st.write(f"Accuracy of the Decision Tree Model: {accuracy * 100:.2f}%")

        st.write("""
        - The accuracy score represents the proportion of correctly classified instances in the test set.
        - A higher accuracy score indicates a better-performing model.
        """)

        st.write("---")  # Separator line

        # Step 8: Simulate Prediction with New Data using Sliders
        st.subheader('Simulated Prediction with User Inputs')

        # Create sliders for user input
        alcohol = st.slider('Alcohol', min_value=10.0, max_value=15.0, value=13.5, step=0.1)
        flavanoids = st.slider('Flavonoids', min_value=1.0, max_value=5.0, value=2.8, step=0.1)
        color_intensity = st.slider('Color Intensity', min_value=1.0, max_value=10.0, value=5.0, step=0.1)

        # Create a DataFrame with the simulated row of data
        simulated_data = pd.DataFrame({
            "alcohol": [alcohol],
            "flavanoids": [flavanoids],
            "color_intensity": [color_intensity]
        })

        # Use the trained decision tree classifier to predict the class of this simulated data
        predicted_class = dtc.predict(simulated_data)

        # Display the predicted class with correct conversion to string
        st.write(f"Predicted class for the simulated data: {str(predicted_class[0])}")

        st.write("""
        - The simulated prediction demonstrates how the trained decision tree model can classify a new data point based on the input features.
        - In this case, the model predicted the wine class for a given set of alcohol, flavanoids, and color intensity values.
        """)

    st.write("---")  # Separator line
    st.write("THE END")
    st.write("© 2025 Dr. Alvin Ang")

# Run the app
if __name__ == "__main__":
    main()

