#note this app only runs locally after installing graphviz and scikit-learn
#on linux i install graphviz via sudo apt-get install graphviz
#go here if u wanna learn how to install into windows
#https://medium.com/python-in-plain-english/how-to-install-graphviz-into-thonny-d8ea9b45b40a

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from graphviz import Source

# Load the iris dataset
iris = pd.read_csv("https://www.alvinang.sg/s/iris_dataset.csv")

# Create a label encoder and transform the species column
le = LabelEncoder()
iris["species"] = le.fit_transform(iris["species"])

# Define features and target variable
X = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = iris["species"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build and train the classifier
dtc = tree.DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train, y_train)

# Streamlit app layout
st.title("Iris Flower Classification")
st.write("Adjust the sliders to predict the class of the Iris flower based on its dimensions.")

# Sliders for user input
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.5)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Create a DataFrame for prediction
simulated_data = pd.DataFrame({
    "sepal_length": [sepal_length],
    "sepal_width": [sepal_width],
    "petal_length": [petal_length],
    "petal_width": [petal_width]
})

# Predict the class of the simulated data
predicted_class = dtc.predict(simulated_data)
predicted_species = le.inverse_transform(predicted_class)

# Display predicted class
st.write(f"Predicted class for the simulated data: **{predicted_species[0]}**")

# Generate and display the decision tree graph
class_names = le.classes_
graph = Source(tree.export_graphviz(dtc, out_file=None,
                      feature_names=X.columns,
                      class_names=class_names,
                      filled=True))
st.subheader("Decision Tree Visualization")
st.graphviz_chart(graph)

