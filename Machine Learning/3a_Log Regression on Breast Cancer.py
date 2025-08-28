import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

#--------------------------------------------------------------------
# Step 1: Load Dataset
#--------------------------------------------------------------------

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/DRALVINANG/Machine-Learning-with-Python-Training/refs/heads/main/Logistic%20Regression/Breast%20Cancer.csv"
    return pd.read_csv(url)

df = load_data()

#--------------------------------------------------------------------
# Step 2: Train-Test Split & Feature Scaling
#--------------------------------------------------------------------

X = df.drop(columns=['Target'])
y = df['Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#--------------------------------------------------------------------
# Step 3: Train Logistic Regression Model
#--------------------------------------------------------------------

logistic_model = LogisticRegression(max_iter=5000)
logistic_model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = logistic_model.predict(X_test_scaled)
y_prob = logistic_model.predict_proba(X_test_scaled)

# Model accuracy
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)

#--------------------------------------------------------------------
# Step 4: Streamlit App Layout
#--------------------------------------------------------------------

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

st.title("🔬 Breast Cancer Prediction using Logistic Regression")
st.write("""
This app predicts whether a breast tumor is **Benign** or **Malignant** based on tumor feature inputs.
It also provides model accuracy, classification metrics, and visual representation of tumor characteristics.
""")

#--------------------------------------------------------------------
# Step 5: Dataset Description
#--------------------------------------------------------------------

st.markdown("---")
st.header("📜 Dataset Description")

st.write("""
The **Breast Cancer Wisconsin Diagnostic Dataset** is used for binary classification, distinguishing between **malignant (cancerous) and benign (non-cancerous) tumors**.

📌 **Dataset Sources:**
- 🔗 [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- 🔗 [Scikit-Learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- 🔗 [GitHub Dataset Link](https://github.com/DRALVINANG/Machine-Learning-with-Python-Training/blob/main/Logistic%20Regression/Breast%20Cancer.csv)

📌 **Features Explained:**
- **Radius:** Mean, standard error, and worst-case radius of the nuclei.
- **Texture:** Variation in pixel intensities in the cell nuclei.
- **Perimeter & Area:** Measurements of the cell nuclei size.
- **Smoothness:** Edge smoothness of the nuclei.
- **Compactness & Concavity:** Shape and extent of concave portions.
- **Concave Points:** Number of concave points in nuclei.
- **Symmetry & Fractal Dimension:** Shape complexity.

📌 **Target Labels:**
- **0:** Malignant (Cancerous)
- **1:** Benign (Non-cancerous)

📌 **Dataset Statistics:**
- 📊 **Total Samples:** 569
- 🏷️ **Classes:** 2 (Malignant, Benign)
- ⚖️ **Class Distribution:**
  - Malignant: 212 (37.3%)
  - Benign: 357 (62.7%)
""")

st.markdown("---")

#--------------------------------------------------------------------
# Step 6: Sidebar - Tumor Feature Inputs for Prediction
#--------------------------------------------------------------------

st.sidebar.header("🎛️ Input Tumor Features for Prediction")

# Define sliders for tumor feature inputs
feature_values = []
feature_names = X.columns

for feature in feature_names:
    min_val = float(X[feature].min())
    max_val = float(X[feature].max())
    default_val = float(X[feature].mean())
    value = st.sidebar.slider(feature, min_val, max_val, default_val)
    feature_values.append(value)

# Convert user input to numpy array and scale it
input_features = np.array(feature_values).reshape(1, -1)
input_features_scaled = scaler.transform(input_features)

# Prediction Button
if st.sidebar.button("🔮 Predict"):
    prediction = logistic_model.predict(input_features_scaled)
    probability = logistic_model.predict_proba(input_features_scaled)[0][1]

    predicted_class = "Benign" if prediction[0] == 1 else "Malignant"
    st.sidebar.success(f"🔍 Predicted Class: **{predicted_class}**")
    st.sidebar.write(f"🧪 Probability of being benign: **{probability:.2f}**")

#--------------------------------------------------------------------
# Step 7: Model Accuracy & Classification Report
#--------------------------------------------------------------------

st.header("📊 Model Accuracy")
st.metric("✅ Accuracy", f"{accuracy * 100:.2f}%")

st.subheader("📌 Classification Report")
st.write(pd.DataFrame(class_report).T)

#--------------------------------------------------------------------
# Step 8: Confusion Matrix
#--------------------------------------------------------------------

st.subheader("📊 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

#--------------------------------------------------------------------
# Step 9: Radar Chart for Tumor Features
#--------------------------------------------------------------------

st.subheader("📡 Radar Chart of Tumor Features")

def plot_radar_chart(features, values):
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    values += values[:1]  # Repeat first value to close the circle
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
    ax.fill(angles, values, color="blue", alpha=0.4)
    ax.plot(angles, values, color="blue", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=9)
    return fig

radar_fig = plot_radar_chart(feature_names.tolist(), feature_values)
st.pyplot(radar_fig)

#--------------------------------------------------------------------
# Step 10: Instructions for Users
#--------------------------------------------------------------------

st.markdown("---")
st.header("📌 How to Use This App")
st.write("""
1. Adjust the **tumor feature sliders** in the sidebar.
2. Click **Predict** to determine if the tumor is **Benign** or **Malignant**.
3. View **model accuracy, classification report, and confusion matrix**.
4. Check the **Radar Chart** to visualize tumor feature distribution.
""")

st.markdown("**Developed by:** Dr. Alvin Ang")

