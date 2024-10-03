import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from graphviz import Source
from sklearn.tree import export_graphviz

#------------------------------------------------------------------------------------------------
# Step 1: Import Dataset
#------------------------------------------------------------------------------------------------
# Load the wine dataset
wine = pd.read_csv("https://www.alvinang.sg/s/wine_sklearn_dataset.csv")

# Streamlit app layout
st.title("Wine Classification")
st.write("Adjust the sliders to predict the class of the wine based on its chemical composition.")

# Show first few rows of the dataset
st.subheader("Wine Dataset")
st.write(wine.head())

#------------------------------------------------------------------------------------------------
# Step 2: Train Test Split
#------------------------------------------------------------------------------------------------
# Target
y = wine["target"]

# Features (All columns except 'target')
X = wine.drop(columns=["target"])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#------------------------------------------------------------------------------------------------
# Step 3: Build and Train the RFC
#------------------------------------------------------------------------------------------------
# Create a RandomForestClassifier object
rfc = RandomForestClassifier(n_estimators=100, criterion="entropy")

# Fit the model to the training data
rfc.fit(X_train, y_train)

#------------------------------------------------------------------------------------------------
# Step 4: Prediction Comparison on Test Data
#------------------------------------------------------------------------------------------------
# Predict the classes for the test data
df = pd.DataFrame({
    "predicted_class": rfc.predict(X_test),
    "actual_class": y_test.tolist()
})

# Display the comparison of predicted and actual class labels
st.subheader("Prediction Comparison on Test Data")
st.write(df)

#------------------------------------------------------------------------------------------------
# Step 5: Accuracy Score
#------------------------------------------------------------------------------------------------
# Calculate and display the accuracy of the model
accuracy = accuracy_score(y_test, rfc.predict(X_test))
st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy * 100:.2f}%")

#------------------------------------------------------------------------------------------------
# Step 6: User Input for Prediction on Simulated Data
#------------------------------------------------------------------------------------------------
# Sliders for user input
st.sidebar.subheader("Input Wine Features")
alcohol = st.sidebar.slider("Alcohol", 11.0, 15.0, 13.0)
malic_acid = st.sidebar.slider("Malic Acid", 0.0, 5.0, 2.0)
ash = st.sidebar.slider("Ash", 1.0, 3.5, 2.5)
alcalinity_of_ash = st.sidebar.slider("Alcalinity of Ash", 10.0, 30.0, 20.0)
magnesium = st.sidebar.slider("Magnesium", 70, 160, 100)
total_phenols = st.sidebar.slider("Total Phenols", 0.0, 4.0, 2.5)
flavanoids = st.sidebar.slider("Flavanoids", 0.0, 6.0, 3.0)
nonflavanoid_phenols = st.sidebar.slider("Nonflavanoid Phenols", 0.0, 1.0, 0.3)
proanthocyanins = st.sidebar.slider("Proanthocyanins", 0.0, 4.0, 1.5)
color_intensity = st.sidebar.slider("Color Intensity", 1.0, 10.0, 4.0)
hue = st.sidebar.slider("Hue", 0.5, 2.0, 1.0)
od280_od315 = st.sidebar.slider("OD280/OD315 of diluted wines", 1.0, 4.0, 2.5)
proline = st.sidebar.slider("Proline", 300, 1700, 750)

# Create a DataFrame with simulated data
simulated_data = pd.DataFrame({
    "alcohol": [alcohol],
    "malic_acid": [malic_acid],
    "ash": [ash],
    "alcalinity_of_ash": [alcalinity_of_ash],
    "magnesium": [magnesium],
    "total_phenols": [total_phenols],
    "flavanoids": [flavanoids],
    "nonflavanoid_phenols": [nonflavanoid_phenols],
    "proanthocyanins": [proanthocyanins],
    "color_intensity": [color_intensity],
    "hue": [hue],
    "od280/od315_of_diluted_wines": [od280_od315],
    "proline": [proline]
})

# Predict the class for the simulated data
predicted_class = rfc.predict(simulated_data)

# Display predicted class
st.subheader("Predicted Class for Simulated Data")
st.write(f"Predicted class: **Class {predicted_class[0]}**")

#------------------------------------------------------------------------------------------------
# Step 7: Visualizing a Tree from the Random Forest
#------------------------------------------------------------------------------------------------
# Visualize the first decision tree in the Random Forest
estimators = rfc.estimators_

# Export and display the first tree in the forest
dot_data = export_graphviz(estimators[0], out_file=None,
                           feature_names=X.columns,
                           class_names=["Class 0", "Class 1", "Class 2"],
                           filled=True, rounded=True)
graph = Source(dot_data)
st.subheader("Decision Tree Visualization")
st.write("(just one randomly chosen tree from the Random Forest...)")
st.graphviz_chart(graph)

