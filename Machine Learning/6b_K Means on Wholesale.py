import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#--------------------------------------------------------------------
# 📦 Step 1: Load the Wholesale Customers Dataset
#--------------------------------------------------------------------

# Title of the App
st.title("🏪 Wholesale Customer Segmentation")
st.write("🔍 Analyzing wholesale customers' purchasing behavior using K-Means clustering.")

st.markdown("---")  # Separator Line

# Load Dataset
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

st.subheader("📄 Dataset Overview")
st.write("This dataset contains annual spending data of wholesale customers across different product categories.")
st.dataframe(df.head())

st.markdown("---")  # Separator Line

#--------------------------------------------------------------------
# 📊 Step 2: Feature Descriptions
#--------------------------------------------------------------------
st.subheader("🔍 Feature Descriptions")
st.write("Understanding the key features in the dataset:")
st.write("- **🏬 Channel**: The type of wholesale customer (1 = Horeca (Hotel/Restaurant/Café), 2 = Retail).")
st.write("- **🌍 Region**: The geographical region of the customer (1 = Lisbon, 2 = Oporto, 3 = Other).")
st.write("- **🥦 Fresh ($)**: Annual spending on fresh products.")
st.write("- **🥛 Milk ($)**: Annual spending on milk products.")
st.write("- **🛒 Grocery ($)**: Annual spending on grocery products.")
st.write("- **❄️ Frozen ($)**: Annual spending on frozen products.")
st.write("- **🧼 Detergents_Paper ($)**: Annual spending on detergents and paper products.")
st.write("- **🥖 Delicassen ($)**: Annual spending on delicatessen products.")

st.markdown("---")  # Separator Line

#--------------------------------------------------------------------
# 🛠 Step 3: Data Preprocessing
#--------------------------------------------------------------------
st.subheader("🛠 Data Preprocessing")

st.write("### 1️⃣ Drop Rows with Any Missing Values")
df = df.dropna()
st.markdown("---")  # Separator Line

st.write("### 2️⃣ Remove Non-Numeric Columns")
df_features = df.drop(columns=['Channel', 'Region'])
st.markdown("---")  # Separator Line

st.write("### 3️⃣ Standardize the Data")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)
st.markdown("---")  # Separator Line

st.subheader("📊 Preprocessed Data")
df_features_display = df_features.copy()
df_features_display = df_features_display.applymap(lambda x: f"${x:,.2f}")
st.dataframe(df_features_display.head())
st.markdown("---")  # Separator Line

#--------------------------------------------------------------------
# 📈 Step 4: Elbow Method to Determine Optimal Clusters
#--------------------------------------------------------------------
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

st.subheader("📐 Elbow Method to Determine Optimal Clusters")
fig, ax = plt.subplots()
ax.plot(K_range, inertia, 'bo-')
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Method")
st.pyplot(fig)
st.write("The Elbow Method helps determine the optimal number of clusters. Look for the 'elbow' point, where the curve bends sharply, as this indicates the best balance between compact clusters and minimal error. Typically, the number of clusters at this bend is chosen for segmentation.")
st.markdown("---")  # Separator Line

#--------------------------------------------------------------------
# 🎯 Step 5: Customer Segments
#--------------------------------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

df_display = df.copy()
df_display[df_features.columns] = df_display[df_features.columns].applymap(lambda x: f"${x:,.2f}")
st.subheader("🎯 Customer Segments")
st.dataframe(df_display)
st.markdown("---")  # Separator Line

#--------------------------------------------------------------------
# 🌐 Step 6: 3D Cluster Visualization
#--------------------------------------------------------------------
st.subheader("🌐 3D Cluster Visualization")
fig = px.scatter_3d(df, x='Fresh', y='Milk', z='Grocery',
                    color=df['Cluster'].astype(str),
                    title='Wholesale Customers Annual Spending Breakdown ($)',
                    labels={'Fresh': 'Fresh Spending ($)', 'Milk': 'Milk Spending ($)', 'Grocery': 'Grocery Spending ($)'},
                    color_discrete_map={'0': 'green', '1': 'red', '2': 'blue'},
                    opacity=0.7)
st.plotly_chart(fig, use_container_width=True)
st.write("### Cluster Insights:")
st.write("- 🟩 **Cluster 0 (Green)**: Low-value customers, who spend the least on fresh, milk, and grocery items.")
st.write("- 🟥 **Cluster 1 (Red)**: Medium-value customers, with moderate spending habits.")
st.write("- 🟦 **Cluster 2 (Blue)**: High-value customers, who spend the most on fresh, milk, and grocery items.")
st.markdown("---")  # Separator Line

#--------------------------------------------------------------------
# 🔮 Step 7: Predict Customer Segment (Sidebar)
#--------------------------------------------------------------------
st.sidebar.header("🔮 Predict Customer Segment")
fresh_spending = st.sidebar.slider("Annual Fresh Products Spending ($)", min_value=0, max_value=100000, value=8000)
milk_spending = st.sidebar.slider("Annual Milk Products Spending ($)", min_value=0, max_value=50000, value=1500)
grocery_spending = st.sidebar.slider("Annual Grocery Products Spending ($)", min_value=0, max_value=100000, value=3000)
frozen = st.sidebar.slider("Annual Frozen Products Spending ($)", min_value=0, max_value=50000, value=2000)
detergents_paper = st.sidebar.slider("Annual Detergents & Paper Spending ($)", min_value=0, max_value=50000, value=600)
delicassen = st.sidebar.slider("Annual Delicassen Spending ($)", min_value=0, max_value=50000, value=1000)

if st.sidebar.button("🔍 Predict Cluster"):
    new_data_scaled = scaler.transform([[fresh_spending, milk_spending, grocery_spending, frozen, detergents_paper, delicassen]])
    predicted_cluster = kmeans.predict(new_data_scaled)[0]
    st.sidebar.success(f"The new customer belongs to Cluster {predicted_cluster}.")

st.markdown("---")  # Separator Line

#--------------------------------------------------------------------
# 🏁 THE END
#--------------------------------------------------------------------
st.write("### 👨‍🏫 Created by Dr. Alvin Ang")

