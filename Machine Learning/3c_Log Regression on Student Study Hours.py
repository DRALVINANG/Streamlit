import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder

#--------------------------------------------------------------------
# Step 1: Set Page Config (Must be First Streamlit Command)
#--------------------------------------------------------------------
st.set_page_config(page_title="Study Hours Prediction", layout="wide")

#--------------------------------------------------------------------
# Step 2: Load Dataset
#--------------------------------------------------------------------

@st.cache_data
def load_data():
    dataset_url = "https://raw.githubusercontent.com/DRALVINANG/Machine-Learning-with-Python-Training/refs/heads/main/Logistic%20Regression/student-study-hours.csv"
    return pd.read_csv(dataset_url)

df = load_data()

# Validate dataset
if df.empty:
    st.error("❌ Error: Dataset failed to load. Please check the URL or internet connection.")
    st.stop()

#--------------------------------------------------------------------
# Step 3: Dataset Description
#--------------------------------------------------------------------

st.title("📚 Student Study Hours & Performance Prediction")
st.write("""
This app demonstrates **Linear & Logistic Regression** to predict student performance  
based on the **number of study hours**.

📌 **Dataset Sources:**
- 🔗 [Kaggle - Student Study Hours](https://www.kaggle.com/datasets/himanshunakrani/student-study-hours)
- 🔗 [GitHub Dataset Link](https://github.com/DRALVINANG/Machine-Learning-with-Python-Training/blob/main/Logistic%20Regression/student-study-hours.csv)

📌 **Dataset Overview:**
- **96 Students** (Data points)
- **Features**: Hours Studied  
- **Targets**: **Scores** (0-100) & **Pass/Fail Classification**  

**Objective:**  
- **Linear Regression** → Predict exam scores based on study hours.  
- **Logistic Regression** → Predict **Pass/Fail** (Score ≥ 50 = Pass).
""")

st.markdown("---")

#--------------------------------------------------------------------
# Step 4: Sidebar - User Inputs for Prediction
#--------------------------------------------------------------------

st.sidebar.header("🎛️ Study Hours & Predictions")

# Interactive Prediction Slider
study_hours = st.sidebar.slider("📖 Select Study Hours:", min_value=0, max_value=10, value=5)

#--------------------------------------------------------------------
# Step 5: Linear Regression - Predicting Scores
#--------------------------------------------------------------------

# Define X and y for Linear Regression
X = df[['Hours']]
y = df['Scores']

# Train Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X, y)

# Predict Scores
y_pred = linear_model.predict(X)

# Find study hours required for a passing score of 50
required_hours = (50 - linear_model.intercept_) / linear_model.coef_[0]

# Predict Score for Given Study Hours
predicted_score = linear_model.predict([[study_hours]])[0]

# Display Linear Regression Prediction
st.sidebar.subheader("📈 Predicted Exam Score")
st.sidebar.success(f"📢 Predicted Score for {study_hours} hours: **{predicted_score:.2f}**")
st.sidebar.write(f"📌 To pass (score ≥ 50), you need to study **{required_hours.item():.2f} hours**.")

#--------------------------------------------------------------------
# Step 6: Logistic Regression - Predicting Pass/Fail
#--------------------------------------------------------------------

# Define Target for Logistic Regression (Pass/Fail)
df['Pass'] = df['Scores'] >= 50
le = LabelEncoder()
y_logistic = le.fit_transform(df['Pass'])

# Train Logistic Regression Model
logistic_model = LogisticRegression()
logistic_model.fit(X, y_logistic)

# Generate Range for Prediction
X_new = np.linspace(df['Hours'].min(), df['Hours'].max(), 300).reshape(-1, 1)
y_logistic_pred_smooth = logistic_model.predict_proba(X_new)[:, 1]

# Find Decision Boundary (Pass/Fail Threshold)
decision_boundary = (-logistic_model.intercept_ / logistic_model.coef_).item()

# Predict Probability of Passing for Given Study Hours
pass_probability = logistic_model.predict_proba([[study_hours]])[0][1]
predicted_pass = "Pass ✅" if pass_probability >= 0.5 else "Fail ❌"

# Display Logistic Regression Prediction
st.sidebar.subheader("📊 Probability of Passing")
st.sidebar.success(f"📢 Predicted Probability: **{pass_probability:.2f}** → **{predicted_pass}**")
st.sidebar.write(f"📌 The decision boundary (transition from fail to pass) occurs at **{decision_boundary:.2f} hours**.")

st.markdown("---")

#--------------------------------------------------------------------
# Step 7: Display Dataset
#--------------------------------------------------------------------

st.subheader("📊 Dataset Preview")
st.write(df.head())

st.markdown("---")

#--------------------------------------------------------------------
# Step 8: Visualization - Linear Regression
#--------------------------------------------------------------------

st.subheader("📈 Linear Regression: Hours Studied vs Scores")
st.write("""
- This model predicts the **expected exam score** based on the number of **hours studied**.
- A **passing score** is 50, marked by the green dashed line.
""")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X, y, color='blue', label='Actual Scores')
ax.plot(X, y_pred, color='red', label=f'Linear Regression\ny = {linear_model.coef_[0]:.2f}x + {linear_model.intercept_:.2f}')
ax.axhline(y=50, color='green', linestyle='--', label='Passing Score = 50')
ax.axvline(x=required_hours.item(), color='purple', linestyle='--', label=f'Pass at {required_hours.item():.2f} Hours')

ax.set_xlabel("Hours Studied")
ax.set_ylabel("Score")
ax.legend()
ax.grid()

st.pyplot(fig)

st.markdown("---")

#--------------------------------------------------------------------
# Step 9: Visualization - Logistic Regression
#--------------------------------------------------------------------

st.subheader("📊 Logistic Regression: Hours vs Pass/Fail Probability")
st.write("""
- This model predicts the **probability of passing** based on study hours.
- A **score ≥ 50** is considered a pass (green region).
""")

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=df['Hours'], y=y_logistic, color='blue', label='Pass/Fail (1 = Pass, 0 = Fail)')
sns.lineplot(x=X_new.flatten(), y=y_logistic_pred_smooth, color='red', label='Logistic Regression Curve')

ax.axhline(y=0.5, color='green', linestyle='--', label='Probability = 0.5')
ax.axvline(x=decision_boundary, color='purple', linestyle='--', label=f'Decision Boundary = {decision_boundary:.2f} Hours')

ax.set_xlabel("Hours Studied")
ax.set_ylabel("Probability of Passing")
ax.legend()
ax.grid()

st.pyplot(fig)

st.markdown("---")

#--------------------------------------------------------------------
# Step 10: Instructions for Users
#--------------------------------------------------------------------

st.header("📌 How to Use This App")
st.write("""
1. **Adjust the study hours slider in the left sidebar** to see the predicted score.
2. **Linear Regression** predicts the **exact score**.
3. **Logistic Regression** predicts **pass/fail probability**.
4. View **graphical insights**:
   - Red **Linear Regression Line** for score trends.
   - Red **Logistic Regression Curve** for pass probability.
   - **Green Line** at score 50 (pass mark).
   - **Purple Decision Boundary** for pass probability (logistic regression).
""")

st.markdown("**Developed by:** Dr. Alvin Ang")

