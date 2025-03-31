import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer

#--------------------------------------------------------------------
# Step 1: Load Dataset for Breast Cancer
#--------------------------------------------------------------------
@st.cache_data
def load_data():
    # Load the breast cancer dataset from sklearn
    data = load_breast_cancer()
    
    # Convert to DataFrame
    df = pd.DataFrame(data['data'], columns=data['feature_names'])
    return df, data['target']

#--------------------------------------------------------------------
# Step 2: Standardize the Data
#--------------------------------------------------------------------
@st.cache_data
def standardize_data(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data

#--------------------------------------------------------------------
# Step 3: Apply PCA
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
    st.title('🌿 Breast Cancer PCA Analysis 🌸')
    st.write("""
    This app performs Principal Component Analysis (PCA) on the Breast Cancer dataset.
    It visualizes the explained variance, the relationship between the first two principal components,
    and the feature loadings for the principal components.
    """)
    st.write("---")

    # Load Dataset
    df, target = load_data()
    st.header('🌸 Breast Cancer Dataset Preview 🌸')
    st.dataframe(df)
    st.markdown("---")

    #--------------------------------------------------------------------
    # Dataset Description
    #--------------------------------------------------------------------
    st.header("📖 Dataset Description")
    st.markdown("""
    - The dataset contains **30 features** that represent various physical and chemical properties of cell nuclei in breast cancer biopsies.
    - The goal is to classify the samples into **malignant (1)** or **benign (0)** categories based on these features.
    - ***UCI:*** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
    - ***SK Learn:*** https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
    - ***own:*** https://github.com/DRALVINANG/Machine-Learning-with-Python-Training/blob/main/Logistic%20Regression/Breast%20Cancer.csv
    """)
    st.markdown("---")
    
    st.markdown("""
    📌 **Breast Cancer Description**
    - The dataset is used for binary classification tasks, specifically to classify breast tumors as either malignant (cancerous) or benign (non-cancerous) based on a series of measurements derived from the digitized images of breast masses collected via fine needle aspiration (FNA) — a common procedure used in cytology to extract cells from breast tissue for microscopic examination..

    📌 **Features (Attributes):**
    - The dataset contains 30 numeric features that are computed from the digitized images of the cells. The features describe various characteristics of the cell nuclei present in the image. These features can be grouped into three categories, each describing different properties of the cell nuclei:
    - **Radius:** Mean, standard error, and worst-case radius of the nuclei.
    - **Texture:** Mean, standard error, and worst-case texture, describing variations in the pixel intensities.
    - **Perimeter:** Mean, standard error, and worst-case perimeter of the nuclei.
    - **Area:** Mean, standard error, and worst-case area of the nuclei.
    - **Smoothness:** Mean, standard error, and worst-case smoothness, which measures how smooth the edges of the nuclei are.
    - **Compactness:** Mean, standard error, and worst-case compactness, which relates the perimeter to the area of the nucleus.
    - **Concavity:** Mean, standard error, and worst-case concavity, measuring the extent of the concave portions of the contour.
    - **Concave Points:** Mean, standard error, and worst-case number of concave points on the nuclei.
    - **Symmetry:** Mean, standard error, and worst-case symmetry of the nuclei.
    - **Fractal Dimension:** Mean, standard error, and worst-case fractal dimension, describing the complexity of the nuclei’s contour.

    📌 **These features are computed in three variants:**
    - **Mean:** The average value of each feature.
    - **Standard Error (SE):** The variation of the feature (how much it deviates from the mean).
    - **Worst Case:** The worst-case (largest) value of each feature.
    - Thus, the 30 features consist of the above categories across three computations (mean, SE, and worst-case).

    📌 **Target (Label):**
    - The target is a binary classification:
    - **0:** Malignant (cancerous)
    - **1:** Benign (non-cancerous)
    - This target indicates whether the tumor is classified as malignant (0) or benign (1).

    📌 **Dataset Composition:**
    - **Number of instances (samples):** 569
    - **Number of features:** 30 numeric features
    - **Number of classes:** 2 (malignant, benign)
    
    📌 **Class distribution:**
    - **Malignant:** 212 instances (37.3%)
    - **Benign:** 357 instances (62.7%)
    
    """)

    st.write("---")

    # Step 1: Standardize Data
    scaled_data = standardize_data(df)

    # Step 2: Apply PCA with 2 Components for Visualization
    st.sidebar.header("🎛️ Select Number of Components")
    n_components = st.sidebar.slider('Select number of components for PCA', min_value=2, max_value=10, value=2, step=1)
    
    pca, pca_result = apply_pca(scaled_data, n_components)

    # Step 3: Scree Plot
    st.header("📊 Scree Plot")
    fig, ax = plt.subplots(figsize=(8, 6))
    PC_values = np.arange(pca.n_components_) + 1
    ax.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    ax.set_title('Scree Plot: Variance Explained by Principal Components')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained')
    st.pyplot(fig)
    
    st.markdown("""
    - We are looking for an Elbow Point to decide an optimal no. of Principal Components (since 10 is too many).
    - (assuming we have 10 components chosen)...We see the Elbow Point either at PC2 or PC3 because the rest of the other Components do not seem to contribute much anymore.

    """)
    st.write("---")

    # Step 4: Cumulative Variance Explained
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
    📌***Assuming 10 components....***
    - We can see from the Cumulative Variance plot that...
    - PC1 contributes 44% to the Target.
    - PC2 contributes 19%. Total is already arouund 65%.
    - When it reaches PC3, there is a significant drop.

    📌***In conclusion,***
    - we can just take PC1 and PC2 and ignore the rest.
    - The remaining (100% — 65% = 35%) are absorbed by the other PCs but they can be disregarded because they are insignificant.
    """)
    st.write("---")

    # Step 5: Visualize PCA Results (Scatter Plot)
    if n_components == 2:
        st.header("🔮 PCA Scatter Plot")
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=target, cmap='plasma')
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_title('PCA: First vs Second Principal Component')
        st.pyplot(fig)
        st.write("---")

    # Step 6: Feature Loadings for Principal Components
    st.header("🔍 Feature Loadings for PCA")
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_components)], index=df.columns)
    st.subheader('Feature Loadings')
    st.dataframe(loadings)

    st.markdown("---")
    # Step 7: Visualize the Loadings
    st.header("📊 Visualizing Feature Loadings")
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_components):
        ax.bar(loadings.index, loadings.iloc[:, i], label=f'PC{i+1}')
    ax.set_xticklabels(loadings.index, rotation=90)
    ax.set_ylabel('Loading Value')
    ax.set_title('Feature Loadings for Principal Components')
    ax.legend()
    st.pyplot(fig)
    st.write("---")

    #--------------------------------------------------------------------
    # Conclusion
    #--------------------------------------------------------------------
    st.markdown("**THE END**")
    st.markdown("© Dr. Your Name")

# Run the app
if __name__ == "__main__":
    main()

