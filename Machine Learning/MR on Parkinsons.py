import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#--------------------------------------------------------------------
# Step 1: Load Dataset
#--------------------------------------------------------------------

@st.cache_data
def load_data():
    dataset_url = "https://raw.githubusercontent.com/DRALVINANG/Machine-Learning-with-Python-Training/refs/heads/main/Multiple%20Regression/Parkinsons.csv"
    data = pd.read_csv(dataset_url)
    
    # Drop missing values & unnecessary columns
    data = data.dropna()
    data = data.drop(columns=['subject#'])  
    return data

# Load dataset
data = load_data()

# Validate dataset
if data.empty:
    st.error("❌ Error: Dataset failed to load. Please check the URL or internet connection.")
    st.stop()

# Select features that match the sidebar inputs
selected_features = ['Jitter(%)', 'Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']

# Define features and target
X = data[selected_features]  # Train only on selected 7 features
y = data['total_UPDRS']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

#--------------------------------------------------------------------
# Step 2: Streamlit App Layout
#--------------------------------------------------------------------

st.set_page_config(page_title="Parkinson's Prediction", layout="wide")

st.title("🧠 Parkinson's Disease Prediction using Linear Regression")
st.write("""
This app predicts the progression severity of Parkinson's disease using **linear regression models**.  
It provides **interactive visualizations, predictions, and model performance metrics** to help users understand the data and model behavior.
""")

st.markdown("---")

#--------------------------------------------------------------------
# Step 3: Sidebar - User Inputs for Prediction
#--------------------------------------------------------------------

st.sidebar.header("🎛️ Input Features for Prediction")

jitter = st.sidebar.slider("📊 Jitter (%)", 0.0, 1.0, 0.01, step=0.01)
shimmer = st.sidebar.slider("📈 Shimmer", 0.0, 1.0, 0.05, step=0.01)
nhr = st.sidebar.slider("🔊 Noise-to-Harmonics Ratio (NHR)", 0.0, 1.0, 0.1, step=0.01)
hnr = st.sidebar.slider("🎵 Harmonics-to-Noise Ratio (HNR)", 0, 50, 20, step=1)
rpde = st.sidebar.slider("🌀 Recurrence Period Density Entropy (RPDE)", 0.0, 1.0, 0.5, step=0.01)
dfa = st.sidebar.slider("🔬 Detrended Fluctuation Analysis (DFA)", 0.0, 1.0, 0.6, step=0.01)
ppe = st.sidebar.slider("🎤 Pitch Period Entropy (PPE)", 0.0, 1.0, 0.2, step=0.01)

# Function to Predict UPDRS Score
def predict_updrs(jitter, shimmer, nhr, hnr, rpde, dfa, ppe):
    input_features = pd.DataFrame([[jitter, shimmer, nhr, hnr, rpde, dfa, ppe]], columns=selected_features)
    predicted_score = model.predict(input_features)[0]
    return predicted_score

# Prediction Button
if st.sidebar.button("🔮 Predict"):
    predicted_updrs = predict_updrs(jitter, shimmer, nhr, hnr, rpde, dfa, ppe)
    st.sidebar.success(f"✅ Predicted Total UPDRS: **{predicted_updrs:.2f}**")

#--------------------------------------------------------------------
# Step 4: Dataset Overview
#--------------------------------------------------------------------

st.header("📂 About the Dataset")

st.write("""
The **Parkinson’s Telemonitoring Dataset** provides biomedical voice measurements for tracking disease progression.

**Features include:**
- **Jitter(%), Shimmer:** Measures of frequency and amplitude variation.
- **NHR, HNR:** Noise-to-harmonics ratios indicating vocal clarity.
- **RPDE, DFA, PPE:** Measures of voice signal unpredictability and variability.
""")

st.subheader("🔍 Dataset Preview")
st.dataframe(data.head())

st.markdown("[📥 Download Dataset](https://raw.githubusercontent.com/DRALVINANG/Machine-Learning-with-Python-Training/refs/heads/main/Multiple%20Regression/Parkinsons.csv)")

st.markdown("---")

#--------------------------------------------------------------------
# Step 5: Generate Visualizations
#--------------------------------------------------------------------

st.header("📊 Data Visualizations")

# Pair Plot
st.subheader("🔗 Pair Plot")
pairplot_fig = sns.pairplot(data[selected_features + ['total_UPDRS']])
st.pyplot(pairplot_fig.fig)

# Correlation Heatmap (Fixed - No Annotations)
st.subheader("🔥 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(data.corr(), annot=False, cmap="coolwarm", fmt=".2f", ax=ax)  # annot=False for clarity
st.pyplot(fig)

st.markdown("---")

#--------------------------------------------------------------------
# Step 6: Model Performance & Residual Plot
#--------------------------------------------------------------------

st.header("📌 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("📉 R² Score", f"{r2:.2f}", help="Measures how well the model fits the data.")
with col2:
    st.metric("📊 Mean Squared Error (MSE)", f"{mse:.2f}", help="Measures prediction accuracy (lower is better).")

st.subheader("📊 Residual Plot (Actual vs Predicted)")
residuals = y_test - y_pred
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(x=y_test, y=residuals, ax=ax)
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("Actual Total UPDRS")
ax.set_ylabel("Residuals")
st.pyplot(fig)

st.markdown("---")

#--------------------------------------------------------------------
# Step 7: Instructions for Users
#--------------------------------------------------------------------

st.header("📌 How to Use This App")
st.write("""
1. Adjust the **biomedical voice measurement sliders** in the sidebar.
2. Click **Predict** to get the estimated **Total UPDRS score**.
3. View the **R² Score & MSE** to understand model performance.
4. Check the **Residual Plot** for error distribution insights.
""")

st.markdown("**Created by:** Dr. Alvin Ang")

