import streamlit as st
import numpy as np
from pyDOE2 import ccdesign
import pandas as pd
import matplotlib.pyplot as plt

st.title("Spectacle Frame Design Experiment using Central Composite Design")

st.write("""
## Central Composite Design (CCD) Overview

Central Composite Design is a popular method in Response Surface Methodology (RSM) for building a second-order (quadratic) model for the response variable.

For our spectacle frame design experiment, we'll consider two factors:
1. Frame Width
2. Temple Length

The CCD will help us explore how these factors affect the comfort and fit of the spectacles, allowing us to optimize the design efficiently.

This experiment uses a Circumscribed Central Composite (CCC) design, which extends beyond the factorial ranges to capture a wider experimental space.

## Code for Generating the Central Composite design using default settings:
\n\ndesign = ccdesign(2)

### Factor Levels:

1. Frame Width (mm):
   - Low (-1): 134.0
   - Center (0): 140.0
   - High (+1): 146.0
   - Axial (-α): 130.0
   - Axial (+α): 150.0

2. Temple Length (mm):
   - Low (-1): 139.0
   - Center (0): 145.0
   - High (+1): 151.0
   - Axial (-α): 135.0
   - Axial (+α): 155.0

Where α = 1.414.

Corner points are combinations of low (-1) and high (+1) levels for both factors.
Axial points explore beyond the factorial space, potentially uncovering optimal conditions outside the initial design space.
Center points (0, 0) are repeated to assess experimental error and check for curvature in the response surface.
""")

# Generate the Central Composite design using default settings (CCC)
design = ccdesign(2)

# Define factor names and their ranges
factors = ["Frame Width (mm)", "Temple Length (mm)"]
ranges = {
    "Frame Width (mm)": (130, 150),  # Typical range for frame width
    "Temple Length (mm)": (135, 155)  # Typical range for temple length
}

# Function to scale the design points to actual values
def scale_point(point, factor):
    low, high = ranges[factor]
    return low + (high - low) * (point + 1.41421356) / (2 * 1.41421356)

# Create a DataFrame
df = pd.DataFrame(design, columns=factors)

# Scale the design points to actual values
for factor in factors:
    df[factor] = df[factor].apply(lambda x: scale_point(x, factor))

# Round the values to 1 decimal place
df = df.round(1)

# Add a run order column
df.insert(0, "Run Order", range(1, len(df) + 1))

# Add a column to identify point types
def identify_point_type(row):
    if row['Run Order'] <= 4:
        return "Factorial (Corner)"
    elif row['Run Order'] <= 8:
        return "Center"
    elif row['Run Order'] <= 12:
        return "Axial"
    else:
        return "Center"

df['Point Type'] = df.apply(identify_point_type, axis=1)

# Create a function to plot the CCD points
def plot_ccd_points(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot points
    factorial = df[df['Point Type'] == 'Factorial (Corner)']
    axial = df[df['Point Type'] == 'Axial']
    center = df[df['Point Type'] == 'Center']
    
    ax.scatter(factorial['Frame Width (mm)'], factorial['Temple Length (mm)'], color='red', s=100, label='Factorial')
    ax.scatter(axial['Frame Width (mm)'], axial['Temple Length (mm)'], color='blue', s=100, label='Axial')
    ax.scatter(center['Frame Width (mm)'].iloc[0], center['Temple Length (mm)'].iloc[0], color='green', s=100, label='Center')
    
    # Add labels
    for _, row in df.iterrows():
        ax.annotate(f"({row['Frame Width (mm)']}, {row['Temple Length (mm)']})", 
                    (row['Frame Width (mm)'], row['Temple Length (mm)']), 
                    xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Frame Width (mm)')
    ax.set_ylabel('Temple Length (mm)')
    ax.set_title('Central Composite Design Points for Spectacle Frame Experiment')
    ax.legend()
    ax.grid(True)
    
    return fig

st.write("## Experiment Design Plan:")
st.dataframe(df)

# Add the visualization
st.write("## Visualization of Design Points:")
fig = plot_ccd_points(df)
st.pyplot(fig)

# Calculate the number of runs for a full 3-level factorial
full_factorial_runs = 3**2  # 3 levels, 2 factors

st.write(f"""
This design includes {len(df)} experimental runs:
- 4 factorial points (corners of the square)
- 4 axial points
- {len(df) - 8} center points for assessing experimental error and checking for curvature in the response.

Comparison with Full Factorial Design:
- Full 3-level factorial design would require {full_factorial_runs} runs.
- This CCD design requires {len(df)} runs.

While the CCD requires more runs than a 3-level factorial for 2 factors, it offers several advantages:
1. Explores points beyond the factorial space (axial points), potentially uncovering optimal conditions outside the initial design space.
2. Allows for efficient estimation of quadratic effects, which is crucial for optimization.
3. Includes center points to assess experimental error and check for curvature in the response surface.
4. For higher numbers of factors, CCD becomes more efficient compared to full factorial designs.

The CCD provides a rich exploration of the design space, balancing the number of runs with the depth of information gained about the response surface.
""")

st.write("""
## Instructions:
1. Conduct each experimental run in the order specified.
2. For each run, create a spectacle frame prototype with the specified Frame Width and Temple Length.
3. Test the prototype for comfort and fit (e.g., using a rating scale or specific measurements).
4. Record the results for each run.
5. After completing all runs, analyze the data to determine the optimal combination of Frame Width and Temple Length.

## Benefits of using CCD for Spectacle Frame Design:
1. Efficiently explores the design space, including points beyond the factorial range.
2. Allows for the estimation of quadratic effects, which can be important for finding optimal comfort and fit.
3. Includes center points to assess experimental error and check for curvature in the response surface.
4. Provides a good balance between experimental effort and information gained about the design space.
5. The CCC design allows for exploration beyond the factorial ranges, potentially uncovering optimal points outside the initial design space.
""")

# Add an option to download the design as a CSV file
csv = df.to_csv(index=False)
st.download_button(
    label="Download experiment design as CSV",
    data=csv,
    file_name="spectacle_frame_ccd_design.csv",
    mime="text/csv",
)

