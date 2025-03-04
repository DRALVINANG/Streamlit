import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from matplotlib.colors import ListedColormap
import plotly.figure_factory as ff

# Load Dataset
@st.cache_data
def load_data():
    url = "https://www.alvinang.sg/s/diabetespima.csv"
    df = pd.read_csv(url)
    df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
                  'DiabetesPedigreeFunction', 'Age', 'Outcome']
    return df

df = load_data()

# Streamlit App Title
st.title("Diabetes Prediction App")
st.write("Predict diabetes using logistic regression on the Pima Indian Diabetes dataset.")

# Dataset Description
st.subheader("Dataset Description")
st.write("""
The **Pima Indian Diabetes dataset** originates from the **National Institute of Diabetes and Digestive and Kidney Diseases**. It is used to predict diabetes in **female patients** of **Pima Indian heritage**, aged **21 years and older**. 

The dataset consists of **768 records** with **8 medical predictor variables** and **1 target variable (Outcome)** indicating the presence of diabetes (**1 = Yes, 0 = No**).

### **Key Features:**
- **Pregnancies** – Number of times pregnant.
- **Glucose** – Plasma glucose concentration after a 2-hour glucose tolerance test.
- **BloodPressure** – Diastolic blood pressure (mm Hg).
- **SkinThickness** – Triceps skinfold thickness (mm), used to estimate body fat.
- **Insulin** – 2-hour serum insulin (mu U/mL), indicating insulin resistance.
- **BMI** – Body mass index (kg/m²), a key risk factor for diabetes.
- **DiabetesPedigreeFunction** – Score estimating genetic predisposition to diabetes.
- **Age** – Age of the patient (years).

### **Target Variable:**
- **Outcome** – **0 (No Diabetes), 1 (Diabetes)**.
""")

# Display Dataset
if st.checkbox("Show raw dataset"):
    st.dataframe(df.head())

# Selecting Features and Outcome
X = df[['Glucose', 'BMI']]
y = df['Outcome']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)

# Sidebar Panel for Inputs
with st.sidebar:
    st.header("User Input")
    glucose_input = st.slider("Glucose", float(df.Glucose.min()), float(df.Glucose.max()), float(df.Glucose.mean()))
    bmi_input = st.slider("BMI", float(df.BMI.min()), float(df.BMI.max()), float(df.BMI.mean()))

    user_input = scaler.transform(np.array([[glucose_input, bmi_input]]))
    prediction = model.predict(user_input)[0]
    st.write(f"### Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")

# Exploratory Data Analysis
st.subheader("Feature Analysis")
feature = st.selectbox("Select feature to visualize:", df.columns[:-1])
fig, ax = plt.subplots()
sns.regplot(x=feature, y='Outcome', data=df, logistic=True, ci=None, y_jitter=0.03, ax=ax)
plt.title(f'{feature} vs Diabetes Outcome')
st.pyplot(fig)

def plot_decision_boundary():
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(('purple', 'green')))
    scatter = ax.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, edgecolor='k', 
                          cmap=ListedColormap(('purple', 'green')))
    plt.xlabel('Glucose (Standardized)')
    plt.ylabel('BMI (Standardized)')
    plt.title('Decision Boundary')
    st.pyplot(fig)

st.subheader("Decision Boundary Visualization")
plot_decision_boundary()

st.markdown("---")
st.subheader("Model Evaluation")

# Accuracy Score
st.markdown("### Accuracy: **{:.2f}**".format(accuracy_score(y_test, model.predict(X_test_scaled))))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, model.predict(X_test_scaled))
fig = ff.create_annotated_heatmap(z=cm, x=["No Diabetes", "Diabetes"], y=["No Diabetes", "Diabetes"], 
                                  colorscale='Blues', showscale=True)
st.subheader("Confusion Matrix")
st.plotly_chart(fig)

# Classification Report Table
st.subheader("Classification Report")
report = classification_report(y_test, model.predict(X_test_scaled), output_dict=True)
df_report = pd.DataFrame(report).transpose()
st.dataframe(df_report.style.format(precision=2))

# Conclusion
st.markdown("---")
st.subheader("Conclusion")
st.write("""
- The model achieved **79% accuracy**, which indicates a reasonably good performance for predicting diabetes.
- The confusion matrix shows **96 true negatives, 25 true positives**, and **some misclassifications**.
- The classification report highlights that precision and recall are **higher for non-diabetic cases** compared to diabetic cases.
- Improvements can be made by using more features, hyperparameter tuning, or alternative machine learning models.
""")
