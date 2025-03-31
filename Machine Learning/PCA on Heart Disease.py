import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px  # For interactive 3D plotting

#--------------------------------------------------------------------
# Step 1: Load Dataset for Heart Disease
#--------------------------------------------------------------------
@st.cache_data
def load_data():
    url = "https://github.com/DRALVINANG/Machine-Learning-with-Python-Training/blob/main/PCA/heart-disease.csv?raw=true"
    df = pd.read_csv(url)
    df['ca'] = pd.to_numeric(df['ca'], errors='coerce')  # Convert to numeric, coerce errors
    df['thal'] = pd.to_numeric(df['thal'], errors='coerce')  # Convert to numeric, coerce errors
    df.dropna(inplace=True)
    return df

#--------------------------------------------------------------------
# Step 2: Separate Features and Target Variable
#--------------------------------------------------------------------
def separate_features_and_target(df):
    X = df.drop(columns='target')  # Assuming 'target' is the column for diagnosis
    y = df['target']
    return X, y

#--------------------------------------------------------------------
# Step 3: Standardize the Data
#--------------------------------------------------------------------
@st.cache_data
def standardize_data(X):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)
    return scaled_data

#--------------------------------------------------------------------
# Step 4: Apply PCA
#--------------------------------------------------------------------
@st.cache_resource
def apply_pca(scaled_data, n_components):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    return pca, pca_result

#--------------------------------------------------------------------
# Main Streamlit App
#--------------------------------------------------------------------
def main():
    st.title('💓 Heart Disease PCA Analysis 💓')
    st.write("""
    This app performs Principal Component Analysis (PCA) on the Heart Disease dataset.
    """)
    st.write("---")

    # Load Dataset
    df = load_data()
    st.header('💡 Heart Disease Dataset Preview 💡')
    st.dataframe(df)
    st.markdown("---")

    # Dataset Description
    st.header("📖 Dataset Description")
    st.markdown("""
    - The dataset contains several features related to heart disease such as age, sex, cholesterol levels, and more.
    - The goal is to classify whether the patient has heart disease based on these features.
    - The target is a binary classification:
        - **0**: No heart disease
        - **1**: Heart disease present
    """)
    st.markdown("---")
    
    st.markdown("""
    - UCI: https://archive.ics.uci.edu/dataset/45/heart+disease
    - own: https://github.com/DRALVINANG/Machine-Learning-with-Python-Training/blob/main/PCA/heart-disease.csv

    📌 **Target Variable: Target (Diagnosis of Heart Disease): This is the target variable that indicates the presence or absence of heart disease.**
    
    📌 **It is often classified as:**

    - 0: No heart disease (healthy)
    - 1, 2, 3, 4: Presence of heart disease (various levels of severity)
    - In many analyses, the target is simplified to a binary classification:
    - 0 (No Disease) and 1 (Disease) by merging the positive classes.

    📌 Features:
    - **age:** Age of the patient (in years). Age is a significant factor, with higher age groups generally being more at risk for heart disease.
    - **sex:** Sex of the patient (1 = male; 0 = female). Men are generally at higher risk of heart disease, though post-menopausal women also face a higher risk.
    - **cp (Chest Pain Type):** Type of chest pain experienced by the patient: 0: Typical angina. 1: Atypical angina. 2: Non-anginal pain. 3: Asymptomatic. Chest pain is a major symptom, with typical angina being a strong indicator of heart disease.
    - **trestbps (Resting Blood Pressure):** Resting blood pressure (in mm Hg) upon admission to the hospital. Higher resting blood pressure can indicate a higher risk of heart disease.
    - **chol** (Serum Cholesterol in mg/dl): Measured cholesterol levels. Elevated cholesterol levels are associated with an increased risk of heart disease.
    - **fbs** (Fasting Blood Sugar > 120 mg/dl): Binary feature indicating whether fasting blood sugar is above 120 mg/dl: 1: True (fasting blood sugar > 120 mg/dl). 0: False (fasting blood sugar ≤ 120 mg/dl). Elevated fasting blood sugar is often an indicator of diabetes, which is a risk factor for heart disease.
    - **restecg** (Resting Electrocardiographic Results): Results of the ECG: 0: Normal. 1: ST-T wave abnormality (e.g., T wave inversions and/or ST elevation or depression of > 0.05 mV). 2: Showing probable or definite left ventricular hypertrophy by Estes’ criteria. ECG results provide crucial information about heart function and rhythm.
    - **thalach** (Maximum Heart Rate Achieved): Maximum heart rate achieved during a stress test. Higher heart rates during exercise typically suggest better cardiovascular health, whereas low maximum heart rate is often associated with heart disease.
    - **exang** (Exercise-Induced Angina): Indicates whether the patient experiences angina during exercise: 1: Yes (angina induced by exercise). 0: No (no angina). Exercise-induced angina is often a symptom of heart disease.
    - **oldpeak** (ST Depression Induced by Exercise Relative to Rest): ST depression value observed during exercise relative to rest. ST depression indicates the likelihood of coronary artery disease. Greater values of ST depression are concerning.
    - **slope** (Slope of the Peak Exercise ST Segment): The slope of the ST segment during peak exercise: 0: Upsloping. 1: Flat. 2: Downsloping. Flat or downsloping ST segments can indicate heart problems, while an upsloping ST segment is usually less concerning.
    - **ca** (Number of Major Vessels Colored by Fluoroscopy): Number of major vessels (ranging from 0 to 3) that are visible under fluoroscopy. A higher number of colored vessels indicates a higher likelihood of heart disease.
    - **thal** (Thalassemia): A blood disorder that affects the body’s ability to carry oxygen: 1: Normal. 2: Fixed defect (no blood flow in part of the heart). 3: Reversible defect (a defect blood flow can be reversed with treatment). Thalassemia is a hereditary condition, but in this context, it indicates heart-related abnormalities.
    """)
    st.markdown("---")

    # Step 1: Separate Features and Target
    X, y = separate_features_and_target(df)

    # Step 2: Standardize Data
    scaled_data = standardize_data(X)

    # Step 3: Apply PCA
    st.sidebar.header("🎛️ Select Number of Components")
    n_components = st.sidebar.slider('Select number of components for PCA', min_value=2, max_value=10, value=3, step=1)
    
    pca, pca_result = apply_pca(scaled_data, n_components)

    # Step 4: Scree Plot
    st.header("📊 Scree Plot")
    fig, ax = plt.subplots(figsize=(8, 6))
    PC_values = np.arange(pca.n_components_) + 1
    ax.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    ax.set_title('Scree Plot: Variance Explained by Principal Components')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained')
    st.pyplot(fig)
    
    st.markdown("""
    - Look for an Elbow Point to decide the optimal number of Principal Components.
    - Typically, if the variance explained starts to drop off significantly after a certain point, you can choose the components before the drop.
    - The elbow point occurs at 2 or 3
    """)
    st.markdown("---")

    # Step 5: Cumulative Variance Explained
    st.header("🔥 Cumulative Variance Explained")
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(PC_values, cumulative_variance, marker='o', linestyle='--', color='green')
    ax.set_title('Cumulative Variance Explained by Principal Components')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Cumulative Variance Explained')
    st.pyplot(fig)
    
    st.subheader("📊 Variance Explained by Each Component")
    explained_variance_ratio = pca.explained_variance_ratio_ * 100  # Convert to percentage
    explained_variance_df = pd.DataFrame({
        'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance_ratio))],
        'Variance Explained (%)': [f"{var:.2f}%" for var in explained_variance_ratio]
    })

    st.dataframe(explained_variance_df)
    st.markdown("""
    - Since the variance of PC1 + 2 + 3 approximately = 45% ++
    - And anything that reaches around 50% is good enough…
    - We can decide to let the elbow point be at PC3… meaning, we select PC1, 2, 3.
    """)
    
    st.markdown("---")
    # Step 6: Visualize PCA Results (3D Scatter Plot)
    if n_components == 3:
        st.header("🔮 PCA 3D Scatter Plot")
        fig = px.scatter_3d(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            z=pca_result[:, 2],
            color=y,  # Color the points by the 'target' variable
            labels={'x': 'First Principal Component', 'y': 'Second Principal Component', 'z': 'Third Principal Component'},
            title='PCA: First vs Second vs Third Principal Component',
            opacity=0.7
        )
        fig.update_traces(marker=dict(size=3))  # Adjust dot size
        st.plotly_chart(fig)
        st.markdown("""
        - in the 3D plot,
        - Darker Colors Represent 3 or 4 - more SEVERE Heart Disease
        - Lighter Colors Represent 0 - likely NO Heart Disease
        """)
        
    st.markdown("---")
    
    # Step 7: Feature Loadings for Principal Components
    st.header("🔍 Feature Loadings for PCA")
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_components)], index=X.columns)
    st.dataframe(loadings)
    
    st.markdown("""
    - **Loadings** represent how much a feature influences each principal component.
    - The loadings closer to 1 (or -1) show a strong influence on that component.
    - Positive Loadings imply that as the factor increases, the principal component also increases.
    - Negative Loadings imply that as the factor increases, the principal component decreases.
    """)

    # Step 8: Visualizing the Loadings
    st.header("📊 Visualizing Feature Loadings")
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_components):
        ax.bar(loadings.index, loadings.iloc[:, i], label=f'PC{i+1}')
    ax.set_xticklabels(loadings.index, rotation=90)
    ax.set_ylabel('Loading Value')
    ax.set_title('Feature Loadings for Principal Components')
    ax.legend()
    st.pyplot(fig)

    st.markdown("""
    📌 **What the colors mean:**
    - Bars represent the loadings for each principal component.
    - The height of each bar shows the magnitude of the feature’s contribution to the component.
    """)

    st.markdown("---")
    
    # Conclusion
    st.markdown("**THE END**")
    st.markdown("© Dr. Alvin Ang")
    st.markdown("---")

# Run the app
if __name__ == "__main__":
    main()

