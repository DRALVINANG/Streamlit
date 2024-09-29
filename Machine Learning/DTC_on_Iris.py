import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from graphviz import Source

# Title of the app
st.title("Iris Flower Species Prediction App")

# Description
st.write("""
This app uses the **Iris dataset** to predict the species of a flower based on its sepal and petal measurements. 
Use the sliders below to adjust the sepal and petal dimensions and see the predicted species in real-time!
""")

# Load the dataset
iris = pd.read_csv("https://www.alvinang.sg/s/iris_dataset.csv")

# Display the dataset
st.write("### Iris Dataset")
st.write(iris)

# Encode the species column
le = LabelEncoder()
iris["species"] = le.fit_transform(iris["species"])

# Target and Features
X = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = iris["species"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the classifier
dtc = tree.DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train, y_train)

# Calculate accuracy
accuracy = accuracy_score(y_test, dtc.predict(X_test))

# Display accuracy score
st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")

# Display the decision tree
st.write("### Decision Tree Visualization")
graph = Source(tree.export_graphviz(dtc, out_file=None,
                                    feature_names=X.columns,
                                    class_names=['Setosa', 'Versicolor', 'Virginica'],
                                    filled=True))
st.graphviz_chart(graph.source)

# Sliders to input sepal and petal dimensions
st.write("### Predict Iris Species Using Sliders")
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Create a DataFrame for the simulated input
simulated_data = pd.DataFrame({
    "sepal_length": [sepal_length],
    "sepal_width": [sepal_width],
    "petal_length": [petal_length],
    "petal_width": [petal_width]
})

# Predict the class for the simulated input
predicted_class = dtc.predict(simulated_data)
predicted_species = le.inverse_transform(predicted_class)[0]

# Display the predicted species
st.write(f"### Predicted Iris Species: {predicted_species}")


