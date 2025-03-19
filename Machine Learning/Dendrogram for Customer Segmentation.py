import streamlit as st
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import pandas as pd

# Function to run the analysis and return relevant info
def run_clustering():
    # Step 1: Load the dataset
    data = {
        'CustomerID': [1, 2, 3, 4, 5],
        'Spending_Groceries': [1000, 1500, 700, 3000, 2000],
        'Spending_Clothes': [200, 400, 150, 600, 500],
        'Spending_Electronics': [300, 800, 250, 900, 1000]
    }
    df = pd.DataFrame(data)
    
    # Step 2: Standardize the Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[['Spending_Groceries', 'Spending_Clothes', 'Spending_Electronics']])

    # Step 3: Generate the Dendrogram
    fig, ax = plt.subplots(figsize=(10, 7))
    dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'), ax=ax)
    
    # Draw a red horizontal line that cuts the dendrogram at the desired height (y=2)
    ax.axhline(y=2, color='r', linestyle='--')

    # Adding more segment lines and number the segments
    for i, d in enumerate(dendrogram['dcoord']):
        ax.plot(d, [i] * len(d), color='k', lw=0.5)  # Line for each segment
        # Annotate the segments with numbers
        ax.text(d[0], i, f'{i + 1}', fontsize=12, verticalalignment='bottom')

    ax.set_title('Dendrogram for Customer Segmentation')
    ax.set_xlabel('Customers')
    ax.set_ylabel('Euclidean Distance')
    st.pyplot(fig)

    # Step 4: Agglomerative Clustering (Hierarchical Clustering)
    cluster = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
    df['Cluster'] = cluster.fit_predict(X_scaled)

    return df

# Streamlit UI elements
st.title('Customer Segmentation with Agglomerative Clustering')

# Move original data to the top
st.subheader('Original Data')
st.write("""
This data contains the spending habits of 5 customers across three categories: Groceries, Clothes, and Electronics.
""")
data = {
    'CustomerID': [1, 2, 3, 4, 5],
    'Spending_Groceries': [1000, 1500, 700, 3000, 2000],
    'Spending_Clothes': [200, 400, 150, 600, 500],
    'Spending_Electronics': [300, 800, 250, 900, 1000]
}
df_original = pd.DataFrame(data)
st.write(df_original)

# Run the clustering analysis
df_clustered = run_clustering()

# Display clustered data
st.subheader('Clustered Data')
st.write(df_clustered[['CustomerID', 'Cluster']])

# Provide interpretation of the clusters
st.subheader('Cluster Interpretation')

st.write("""
- **Cluster 0**: Customers 1 and 4: Premium customers with high expenditures across groceries and electronics, and moderate-to-high spending on clothes.
- **Cluster 1**: Customers 0 and 2: Low spending profile, particularly on clothes and electronics.
- **Cluster 2**: Customer 3: A high spender across all categories, possibly an outlier or high-income customer.
""")

# Explanation of the dendrogram and the red line
st.subheader('Dendrogram Explanation')

st.write("""
### What is a Dendrogram?
A dendrogram is a tree-like diagram that records the sequences of merges in hierarchical clustering. It helps visualize how clusters are formed and how similar the data points are to one another.

- **The Red Line**: The red horizontal line at y=2 is the threshold where we "cut" the dendrogram to form clusters. The idea is to draw the line at a level where clusters are distinct enough but not too far apart. 
- **The Distance**: The height at which the red line cuts represents the distance between clusters. A higher cut means that clusters are more distinct (fewer but more homogeneous groups), and a lower cut means that more clusters are formed, but they may not be as distinct.
- **Cluster Formation**: Each vertical line in the dendrogram represents a cluster of customers. The longer the line, the more distinct the two clusters are. When you draw the red line, the vertical lines below the cut represent the clusters.

### Cluster Assignments
- **Cluster 0**: High spenders (Customers 1 & 4)
- **Cluster 1**: Low spenders (Customers 0 & 2)
- **Cluster 2**: Very high spender (Customer 3)
""")

# Further explanation of the methodology
st.subheader('Explanation of the Methodology')
st.write("""
### Linkage & Method
- **Linkage**: Measures how the distance between two clusters is calculated. Ward's method is used here because it minimizes within-cluster variance.
- **Metric**: We use Euclidean distance to measure the distance between data points.
- **Agglomerative Clustering**: This is a bottom-up approach where each data point starts as its own cluster and merges with the nearest clusters until the desired number of clusters is achieved.

### Agglomerative Clustering
Agglomerative clustering works by iteratively merging the closest pairs of clusters. The process is visualized in the dendrogram, which shows how clusters are progressively merged based on their similarity.
""")

# Displaying further reference
st.write("""
For more information, you can refer to [Clustering Methods Reference](https://www.alvinang.sg/s/Clustering_methods_2023.pdf).

Created by Dr. Alvin Ang
""")
