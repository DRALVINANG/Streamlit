import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import plotly.express as px

# Function to run clustering analysis and return results
def run_clustering():
    # Load dataset
    url = 'https://raw.githubusercontent.com/DRALVINANG/Machine-Learning-with-Python-Training/main/Hierarchical%20Clustering/Mall%20Customers.csv'
    df = pd.read_csv(url)

    # Select features
    X = df[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']]

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Generate dendrogram
    fig, ax = plt.subplots(figsize=(10, 7))
    dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'), ax=ax)
    ax.axhline(y=7, color='r', linestyle='--')
    ax.set_title('Dendrogram for Customer Segmentation')
    ax.set_xlabel('Customers')
    ax.set_ylabel('Euclidean Distance')
    st.pyplot(fig)

    # Perform clustering
    cluster = AgglomerativeClustering(n_clusters=6, metric='euclidean', linkage='ward')
    df['Cluster'] = cluster.fit_predict(X_scaled)

    return df

# Streamlit App
st.title('💸MALL Customers Segmentation using Hierarchical Clustering')
st.markdown("---")

# Original data section
st.header('✅Original Data')
st.write("""
This dataset contains information about mall customers, including their age, annual income, and spending score.
""")
url = 'https://raw.githubusercontent.com/DRALVINANG/Machine-Learning-with-Python-Training/main/Hierarchical%20Clustering/Mall%20Customers.csv'
df_original = pd.read_csv(url)
st.write(df_original)
st.markdown("---")

#Dataset Explanation
st.header('About the Dataset')
st.write("""
📌**Kaggle:** https://www.kaggle.com/datasets/shwetabh123/mall-customers

📌 **Own:** https://www.alvinang.sg/s/hierarchical-clustering-with-python-and-scikit-learn-shopping-data.csv

📌 This dataset contains data about 200 customers of a mall, with the following features:
- **CustomerID:** Unique identifier for each customer.
- **Genre:** Gender of the customer (Male/Female).
- **Age:** The age of the customer.
- **Annual Income (k$):** The customer’s annual income in thousands of dollars.
- **Spending Score (1–100):** A score assigned by the mall based on customer behavior and spending patterns (with 1 being low spending and 100 being high spending).

📌This dataset is typically used for unsupervised learning tasks like clustering, which aims to segment customers into different groups based on shared characteristics.
""")
st.markdown("---")


# Run clustering
df_clustered = run_clustering()
st.markdown("---")

# Methodology explanation
st.header('📜Explanation of the Methodology')
st.write("""
### Linkage & Method
- **Linkage**: Ward's method is used to minimize within-cluster variance.
- **Metric**: Euclidean distance measures the distance between data points.
- **Agglomerative Clustering**: A bottom-up approach where each point starts as its own cluster and clusters merge step-by-step.

### Dendrogram Interpretation
- The dendrogram illustrates how data points are merged.
- The red line at y=7 is the threshold used to determine the number of clusters (6 in this case).
""")
st.markdown("---")


# Clustered data output
st.header('📈Clustered Data')
st.write(df_clustered[['CustomerID', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']].head())
st.markdown("---")

# Interactive 3D visualization
st.header('📊3D Visualization of Clusters')
fig_3d = px.scatter_3d(df_clustered,
                       x='Annual Income (k$)',
                       y='Spending Score (1-100)',
                       z='Age',
                       color='Cluster',
                       title='Customer Segments Based on Hierarchical Clustering (3D)',
                       symbol='Cluster',
                       opacity=0.8)
st.plotly_chart(fig_3d)
st.markdown("---")

# Additional reference
st.write("""
For more information, you can refer to [Clustering Methods Reference](https://www.alvinang.sg/s/Clustering_methods_2023.pdf).

🤓 Created by Dr. Alvin Ang
""")
st.markdown("---")
