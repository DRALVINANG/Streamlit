import streamlit as st
import numpy as np
from pyDOE2 import bbdesign
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations, product

st.title("Handphone Cover Experiment Design")

st.write("""
For our handphone cover experiment, let's consider these three factors:
- Material thickness (mm):
\n			-1 (thin) / 0 (medium) / 1 (thick)
- Surface texture:
\n			-1 (smooth) / 0 (slightly textured) / 1 (rough)
- Edge design:
\n			-1 (flat) / 0 (slightly curved) / 1 (rounded)

The Box-Behnken design will help us efficiently explore how these factors affect the phone cover's performance, such as drop protection or user comfort, without testing every possible combination.
""")

# Generate the Box-Behnken design
design = bbdesign(3)

# Create factor labels
factors = ["Material Thickness", "Surface Texture", "Edge Design"]

# Create a DataFrame
df = pd.DataFrame(design, columns=factors)

# Replace -1, 0, 1 with actual factor levels
thickness_map = {-1: "Thin", 0: "Medium", 1: "Thick"}
texture_map = {-1: "Smooth", 0: "Slightly Textured", 1: "Rough"}
edge_map = {-1: "Flat", 0: "Slightly Curved", 1: "Rounded"}

df["Material Thickness"] = df["Material Thickness"].map(thickness_map)
df["Surface Texture"] = df["Surface Texture"].map(texture_map)
df["Edge Design"] = df["Edge Design"].map(edge_map)

# Add a run order column
df.insert(0, "Run Order", range(1, len(df) + 1))

st.write("Experiment Design Plan:")
st.dataframe(df)

st.write("This design includes 15 experimental runs, including 3 center points for assessing experimental error and checking for curvature in the response.")

st.write("Instructions:")
st.write("1. Conduct each experimental run in the order specified.")
st.write("2. For each run, create a phone cover with the specified characteristics.")
st.write("3. Test the cover for the desired response (e.g., drop protection, user comfort).")
st.write("4. Record the results for each run.")
st.write("5. After completing all runs, analyze the data to determine the optimal combination of factors.")

def plot_box_behnken_design():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Box-Behnken design points
    points = np.array([
        [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 1, 0],
        [-1, 0, -1], [1, 0, -1], [-1, 0, 1], [1, 0, 1],
        [0, -1, -1], [0, 1, -1], [0, -1, 1], [0, 1, 1],
        [0, 0, 0], [0, 0, 0], [0, 0, 0]
    ])
    
    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', s=100)
    
    # Plot the cube
    r = [-1, 1]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="b")
    
    ax.set_xlabel('Material Thickness')
    ax.set_ylabel('Surface Texture')
    ax.set_zlabel('Edge Design')
    ax.set_title('Box-Behnken Design Points')
    
    return fig

st.write("Visualization of Box-Behnken Design Points:")
fig = plot_box_behnken_design()
st.pyplot(fig)
