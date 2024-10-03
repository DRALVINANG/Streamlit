import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Function to draw radar (spider) chart
def plot_radar_chart(feature_values, feature_labels):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    theta = np.linspace(0, 2 * np.pi, len(feature_values), endpoint=False).tolist()
    
    # Complete the loop for the radar chart
    feature_values += feature_values[:1]
    theta += theta[:1]
    
    ax.fill(theta, feature_values, 'b', alpha=0.1)
    ax.plot(theta, feature_values, 'b', label="Feature Values")
    
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(feature_labels, fontsize=8)
    ax.set_yticklabels([])
    plt.title('Radar Chart of Tumor Features')
    st.pyplot(fig)

#------------------------------------------------------------------------------------------------
# Step 1: Load Dataset
#------------------------------------------------------------------------------------------------
# Load the Breast Cancer dataset from the provided GitHub link
url = "https://raw.githubusercontent.com/DRALVINANG/Machine-Learning-with-Python-Training/refs/heads/main/Logistic%20Regression/Breast%20Cancer.csv"
df = pd.read_csv(url)

# Streamlit app layout
st.title("Breast Cancer Classification")
st.write("Predict if a tumor is malignant or benign based on various features.")

# Show first few rows of the dataset
st.subheader("Breast Cancer Dataset")
st.write(df.head())

#------------------------------------------------------------------------------------------------
# Step 2: Train-Test Split
#------------------------------------------------------------------------------------------------
# Define X (features) and y (target)
X = df.drop(columns=['Target'])
y = df['Target']

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

#------------------------------------------------------------------------------------------------
# Step 3: Logistic Regression Model
#------------------------------------------------------------------------------------------------
# Initialize and fit the logistic regression model
logistic_model = LogisticRegression(max_iter=5000)  # Increase max_iter to ensure convergence
logistic_model.fit(X_train, y_train)

# Predictions on the test set
y_pred = logistic_model.predict(X_test)

#------------------------------------------------------------------------------------------------
# Step 4: Model Evaluation with Confusion Matrix
#------------------------------------------------------------------------------------------------
# Display Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
st.pyplot(fig)

# Calculate and display Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy * 100:.2f}%")

# Display classification report
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

#------------------------------------------------------------------------------------------------
# Step 5: User Input for Prediction on Simulated Data
#------------------------------------------------------------------------------------------------
# Sidebar sliders for user input on various features
st.sidebar.subheader("Input Tumor Features for Prediction")

# Creating sliders for the user to input features
mean_radius = st.sidebar.slider("Mean Radius", float(X['mean radius'].min()), float(X['mean radius'].max()), float(X['mean radius'].mean()))
mean_texture = st.sidebar.slider("Mean Texture", float(X['mean texture'].min()), float(X['mean texture'].max()), float(X['mean texture'].mean()))
mean_perimeter = st.sidebar.slider("Mean Perimeter", float(X['mean perimeter'].min()), float(X['mean perimeter'].max()), float(X['mean perimeter'].mean()))
mean_area = st.sidebar.slider("Mean Area", float(X['mean area'].min()), float(X['mean area'].max()), float(X['mean area'].mean()))
mean_smoothness = st.sidebar.slider("Mean Smoothness", float(X['mean smoothness'].min()), float(X['mean smoothness'].max()), float(X['mean smoothness'].mean()))
mean_compactness = st.sidebar.slider("Mean Compactness", float(X['mean compactness'].min()), float(X['mean compactness'].max()), float(X['mean compactness'].mean()))
mean_concavity = st.sidebar.slider("Mean Concavity", float(X['mean concavity'].min()), float(X['mean concavity'].max()), float(X['mean concavity'].mean()))
mean_concave_points = st.sidebar.slider("Mean Concave Points", float(X['mean concave points'].min()), float(X['mean concave points'].max()), float(X['mean concave points'].mean()))
mean_symmetry = st.sidebar.slider("Mean Symmetry", float(X['mean symmetry'].min()), float(X['mean symmetry'].max()), float(X['mean symmetry'].mean()))
mean_fractal_dimension = st.sidebar.slider("Mean Fractal Dimension", float(X['mean fractal dimension'].min()), float(X['mean fractal dimension'].max()), float(X['mean fractal dimension'].mean()))

# Generate the input data for prediction based on user input
input_data = pd.DataFrame({
    'mean radius': [mean_radius],
    'mean texture': [mean_texture],
    'mean perimeter': [mean_perimeter],
    'mean area': [mean_area],
    'mean smoothness': [mean_smoothness],
    'mean compactness': [mean_compactness],
    'mean concavity': [mean_concavity],
    'mean concave points': [mean_concave_points],
    'mean symmetry': [mean_symmetry],
    'mean fractal dimension': [mean_fractal_dimension],
    # Adding default values for other required features
    'radius error': [X['radius error'].mean()],
    'texture error': [X['texture error'].mean()],
    'perimeter error': [X['perimeter error'].mean()],
    'area error': [X['area error'].mean()],
    'smoothness error': [X['smoothness error'].mean()],
    'compactness error': [X['compactness error'].mean()],
    'concavity error': [X['concavity error'].mean()],
    'concave points error': [X['concave points error'].mean()],
    'symmetry error': [X['symmetry error'].mean()],
    'fractal dimension error': [X['fractal dimension error'].mean()],
    'worst radius': [X['worst radius'].mean()],
    'worst texture': [X['worst texture'].mean()],
    'worst perimeter': [X['worst perimeter'].mean()],
    'worst area': [X['worst area'].mean()],
    'worst smoothness': [X['worst smoothness'].mean()],
    'worst compactness': [X['worst compactness'].mean()],
    'worst concavity': [X['worst concavity'].mean()],
    'worst concave points': [X['worst concave points'].mean()],
    'worst symmetry': [X['worst symmetry'].mean()],
    'worst fractal dimension': [X['worst fractal dimension'].mean()],
})

# Predict the class for the input data
predicted_class = logistic_model.predict(input_data)
predicted_probability = logistic_model.predict_proba(input_data)

# Display predicted class and probability
st.subheader("Prediction for Simulated Data")
st.write(f"Predicted class: **{'Benign' if predicted_class[0] == 1 else 'Malignant'}**")
st.write(f"Probability of being benign: **{predicted_probability[0][1]:.2f}**")

#------------------------------------------------------------------------------------------------
# Step 6: Display Radar Chart
#------------------------------------------------------------------------------------------------
# Create a radar chart to visualize feature input
plot_radar_chart(
    feature_values=[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, 
                    mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension],
    feature_labels=['Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness', 
                    'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry', 'Mean Fractal Dimension']
)
