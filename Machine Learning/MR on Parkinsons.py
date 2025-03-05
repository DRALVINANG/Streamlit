import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm

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
# Step 3: About the Dataset
#--------------------------------------------------------------------

st.header("📜 About the Dataset")

st.write("""
The **Parkinson’s Telemonitoring Dataset** is used to monitor the progression of **Parkinson’s disease**  
based on **biomedical voice measurements**. It is a **regression dataset** where the goal is to predict  
the **severity of the disease** using these features.

📌 **Dataset Sources:**
- 🔗 [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring)
- 🔗 [GitHub Dataset Link](https://github.com/DRALVINANG/Machine-Learning-with-Python-Training/blob/main/Multiple%20Regression/Parkinsons.csv)

📌 **Key Features:**
- **Jitter(%)**: Measures variation in frequency, capturing frequency instability.
- **Jitter (Abs, RAP, PPQ5, DDP)**: Different calculations of jitter representing irregularities in vocal frequency.
- **Shimmer**: Measures **variation in amplitude**, showing fluctuations in loudness.
- **Shimmer (dB, APQ3, APQ5, APQ11, DDA)**: Different variations of shimmer capturing amplitude instability.
- **NHR (Noise-to-Harmonics Ratio)**: Indicates **the level of noise** in the voice relative to vocal sound.
- **HNR (Harmonics-to-Noise Ratio)**: Measures **voice clarity**, representing the balance of harmonic sounds and noise.
- **RPDE (Recurrence Period Density Entropy)**: A **nonlinear dynamic feature** that measures **unpredictability** in the voice signal.
- **DFA (Detrended Fluctuation Analysis)**: Analyzes the **self-similarity** of voice signals over time.
- **PPE (Pitch Period Entropy)**: Measures **pitch variability**, showing how much the pitch period changes over time.

📌 **Target Variables (What We Predict):**
- **Motor UPDRS**: A score reflecting the severity of motor-related symptoms in Parkinson’s disease.
- **Total UPDRS**: A score reflecting **overall disease severity**, including both **motor and non-motor** symptoms.

📌 **How These Features Affect Disease Severity:**
- **Higher jitter, shimmer, and NHR** → **More severe Parkinson’s disease**
- **Lower HNR and higher RPDE** → **Less vocal clarity, indicating disease progression**
- **Higher DFA & PPE** → **More irregularity in voice, linked to severe cases**
""")

st.markdown("---")

#--------------------------------------------------------------------
# Step 4: Sidebar - User Inputs for Prediction
#--------------------------------------------------------------------

st.sidebar.header("🎛️ Input Features for Prediction")

jitter = st.sidebar.slider("📊 Jitter (%)", 0.0, 1.0, 0.01, step=0.01)
shimmer = st.sidebar.slider("📈 Shimmer", 0.0, 1.0, 0.05, step=0.01)
nhr = st.sidebar.slider("🔊 Noise-to-Harmonics Ratio (NHR)", 0.0, 1.0, 0.1, step=0.01)
hnr = st.sidebar.slider("🎵 Harmonics-to-Noise Ratio (HNR)", 0, 50, 20, step=1)
rpde = st.sidebar.slider("🌀 Recurrence Period Density Entropy (RPDE)", 0.0, 1.0, 0.5, step=0.01)
dfa = st.sidebar.slider("🔬 Detrended Fluctuation Analysis (DFA)", 0.0, 1.0, 0.6, step=0.01)
ppe = st.sidebar.slider("🎤 Pitch Period Entropy (PPE)", 0.0, 1.0, 0.2, step=0.01)

# Prediction Function
def predict_updrs(jitter, shimmer, nhr, hnr, rpde, dfa, ppe):
    input_features = pd.DataFrame([[jitter, shimmer, nhr, hnr, rpde, dfa, ppe]], columns=selected_features)
    predicted_score = model.predict(input_features)[0]
    return predicted_score

# Prediction Button
if st.sidebar.button("🔮 Predict"):
    predicted_updrs = predict_updrs(jitter, shimmer, nhr, hnr, rpde, dfa, ppe)
    st.sidebar.success(f"✅ Predicted Total UPDRS: **{predicted_updrs:.2f}**")

# Add a horizontal line to segment the prediction and model performance
st.sidebar.markdown("---")

#--------------------------------------------------------------------
# Step 5: Model Performance
#--------------------------------------------------------------------

st.sidebar.header("📊 Model Performance")

# Model Performance Metrics
st.sidebar.metric("📉 R² Score", f"{r2:.2f}")
st.sidebar.metric("📊 Mean Squared Error (MSE)", f"{mse:.2f}")

# Provide a description of R² and MSE
st.sidebar.write("""
**R² Score**:
- R² represents the proportion of variance in the target variable that can be explained by the model.
- **A good R² score** is generally **above 0.7**. A score close to **1** indicates that the model fits the data well, while **below 0.3** indicates a poor fit.

**Mean Squared Error (MSE)**:
- MSE measures the average squared difference between actual and predicted values. **Lower values** indicate better model performance.
- **Good MSE** values are **closer to 0**.
""")

st.markdown("---")

#--------------------------------------------------------------------
# Step 6: Generate Visualizations
#--------------------------------------------------------------------

st.header("📊 Data Visualizations")

# **Pair Plot** for selected features
st.subheader("🔗 Pair Plot for Selected Features")
fig = sns.pairplot(data[selected_features + ['total_UPDRS']])
st.pyplot(fig)

st.markdown("""
**Pair Plot Description:**
- The pair plot shows the relationships between pairs of features (e.g., Jitter(%) vs. Shimmer, RPDE vs. NHR) and how they relate to the target variable (Total UPDRS).
- **Jitter(%) and Shimmer** show a potential positive correlation with **Total UPDRS**.
- **RPDE and NHR** also seem to show a positive relationship with **Total UPDRS**, indicating that higher values of RPDE and NHR could be linked to more severe disease progression.
- **Total UPDRS** is visualized on the diagonal, showing the distribution of the target variable.
""")

st.markdown("---")

#--------------------------------------------------------------------
# Step 7: Feature Importance using MR
#--------------------------------------------------------------------
st.subheader("🔍 Feature Importance using Multiple Regression")

# Add constant to the features for OLS regression
X_ols = sm.add_constant(X)

# Fit OLS Model
ols_model = sm.OLS(y, X_ols).fit()

# Extract p-values from the OLS model
p_values = ols_model.pvalues[1:]  # exclude constant
p_values_sorted = p_values.sort_values(ascending=True)

# Create a bar plot of 1 - P-value for feature importance
fig, ax = plt.subplots(figsize=(5, 5))
sns.barplot(x=1 - p_values_sorted, y=p_values_sorted.index, palette='Set2', ax=ax)

# Add a red dashed line for the 0.95 threshold
plt.axvline(x=0.95, color='r', linestyle='dotted')

# Annotate the threshold
plt.annotate('0.95', xy=(0.95, 2.5), xycoords='data', color='r')

# Add labels and title
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Visualizing Important Features using Multiple Regression')
st.pyplot(fig)

st.markdown("""
### 📊 Feature Importance Plot
This chart visualizes the importance of each feature in predicting the target variable (Total UPDRS) using Ordinary Least Squares (OLS) regression. The bars represent the feature importance scores, derived from the p-values. A **larger score** indicates that the feature has a **stronger and more statistically significant relationship** with the target variable.

The **red dashed line at 0.95** represents the typical significance threshold. Features with **scores above 0.95** are considered **significant**, indicating they have a strong relationship with the target variable, and thus are more important in predicting disease severity. Features with scores **below 0.95** have a weaker influence on disease progression and may not be as important for the model's prediction.
""")

st.markdown("---")

#--------------------------------------------------------------------
# Step 8: Residual Plot (Actual vs Predicted)
#--------------------------------------------------------------------

st.header("📊 Residual Plot (Actual vs Predicted)")

residuals = y_test - y_pred
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(x=y_test, y=residuals, ax=ax)
ax.axhline(0, color="red", linestyle="--", linewidth=1)
st.pyplot(fig)

st.markdown("""
**Residual Plot Description:**
- The residual plot shows the **errors between actual and predicted values**.
- The **red dashed line** represents the **ideal model**, where there are no residuals (errors).
- **If the points are randomly spread around the red line**, the model is likely a good fit.
- **If the points show a pattern**, such as clustering or a clear trend, it indicates that the model does not fit well, and **Multiple Regression (MR)** may not be the best model for this dataset.
""")

st.markdown("---")

#--------------------------------------------------------------------
# Step 9: How to Use the App
#--------------------------------------------------------------------

st.markdown("### 📌 How to Use This App")
st.write("""
1. Adjust the **features** (jitter, shimmer, NHR, HNR, RPDE, DFA, PPE) using the sliders in the sidebar.
2. Click on **Predict** to calculate the **predicted UPDRS score** for the disease progression.
3. View **data visualizations** (correlation heatmap, feature importance) and **model performance metrics**.
""")

# Add a horizontal line to indicate the end of the app
st.markdown("---")
st.markdown("**THE END**")
st.markdown("© Dr. Alvin Ang")

