import streamlit as st
import numpy as np
from pyDOE2 import ccdesign
import pandas as pd

st.title("Customizable Central Composite Design (CCD) Generator")

st.write("""
## Central Composite Design (CCD) Overview

Central Composite Design is a popular method in Response Surface Methodology (RSM) for building a second-order (quadratic) model for the response variable.

This app allows you to create a customized Circumscribed Central Composite (CCC) design for 2 to 10 factors.
""")

# Let user choose the number of factors
num_factors = st.slider("Select the number of factors:", min_value=2, max_value=10, value=2)

# Create input fields for factor names and levels
factors = []
ranges = {}

for i in range(num_factors):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        factor_name = st.text_input(f"Factor {i+1} name:", value=f"Factor {i+1}")
    
    with col2:
        low_value = st.number_input(f"{factor_name} low value:", value=0.0, format="%.2f")
    
    with col3:
        high_value = st.number_input(f"{factor_name} high value:", value=1.0, format="%.2f")
    
    factors.append(factor_name)
    ranges[factor_name] = (low_value, high_value)

# Generate the Central Composite design
if st.button("Generate CCD Design"):
    design = ccdesign(num_factors)

    # Function to scale the design points to actual values
    def scale_point(point, factor):
        low, high = ranges[factor]
        return low + (high - low) * (point + 1.41421356) / (2 * 1.41421356)

    # Create a DataFrame
    df = pd.DataFrame(design, columns=factors)

    # Scale the design points to actual values
    for factor in factors:
        df[factor] = df[factor].apply(lambda x: scale_point(x, factor))

    # Round the values to 2 decimal places
    df = df.round(2)

    # Add a run order column
    df.insert(0, "Run Order", range(1, len(df) + 1))

    # Add a column to identify point types
    def identify_point_type(row):
        if row['Run Order'] <= 2**num_factors:
            return "Factorial"
        elif row['Run Order'] <= 2**num_factors + 2*num_factors:
            return "Axial"
        else:
            return "Center"

    df['Point Type'] = df.apply(identify_point_type, axis=1)

    st.write("## Experiment Design Plan:")
    st.dataframe(df)

    st.write("## Factor Levels:")
    for factor in factors:
        low, high = ranges[factor]
        center = (low + high) / 2
        axial_low = low - 0.41421356 * (high - low) / 2
        axial_high = high + 0.41421356 * (high - low) / 2
        
        st.write(f"""
        {factor}:
        - Low (-1): {low:.2f}
        - Center (0): {center:.2f}
        - High (+1): {high:.2f}
        - Axial (-α): {axial_low:.2f}
        - Axial (+α): {axial_high:.2f}
        """)

    st.write(f"""
    Where α = 1.414 for a rotatable design.

    This design includes {len(df)} experimental runs:
    - {2**num_factors} factorial points
    - {2*num_factors} axial points
    - {len(df) - 2**num_factors - 2*num_factors} center points for assessing experimental error and checking for curvature in the response.
    """)

    st.write("""
    ## Instructions:
    1. Conduct each experimental run in the order specified.
    2. For each run, set up your experiment with the specified factor levels.
    3. Measure and record the response variable(s) for each run.
    4. After completing all runs, analyze the data to determine the optimal combination of factors.
    """)

    # Add an option to download the design as a CSV file
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download experiment design as CSV",
        data=csv,
        file_name="ccd_design.csv",
        mime="text/csv",
    )

