import streamlit as st
import numpy as np
import pandas as pd
from pyDOE2 import pbdesign
import base64

def get_csv_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="plackett_burman_design.csv">Download CSV File</a>'
    return href

st.title("Custom Plackett-Burman Design Generator")

st.write("""
## About Plackett-Burman Designs
Plackett-Burman designs are two-level fractional factorial designs used for screening experiments. 
They allow researchers to identify the most important factors affecting a process or product with a 
minimal number of experimental runs.
""")

# Allow user to select number of factors
n_factors = st.slider("Select number of factors", min_value=8, max_value=16, value=8, step=1)

# Create input fields for factor names and levels
factor_names = []
factor_levels = {}

st.write("## Enter Factor Names and Levels")
st.write("For each factor, enter the name and the two levels (low and high).")

for i in range(n_factors):
    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.text_input(f"Factor {i+1} Name", value=f"Factor {i+1}")
    with col2:
        low = st.text_input(f"Factor {i+1} Low Level (-1)", value="Low")
    with col3:
        high = st.text_input(f"Factor {i+1} High Level (+1)", value="High")
    
    factor_names.append(name)
    factor_levels[name] = (low, high)

if st.button("Generate Plackett-Burman Design"):
    # Generate the Plackett-Burman design
    design = pbdesign(n_factors)

    st.write("## Plackett-Burman Design Matrix")
    design_df = pd.DataFrame(design, columns=factor_names)
    st.dataframe(design_df)

    # Add download button
    st.markdown(get_csv_download_link(design_df), unsafe_allow_html=True)

    st.write("## Factors and Levels")
    for factor, (low, high) in factor_levels.items():
        st.write(f"- **{factor}**: {low} (-1), {high} (+1)")

    st.write(f"Number of runs in Plackett-Burman design: {len(design)}")
    st.write(f"Number of runs in full factorial design: {2**n_factors}")
    st.write(f"Number of runs saved: {2**n_factors - len(design)}")

    st.write("""
    ## Next Steps
    1. Conduct the experiments according to this design.
    2. Analyze the results to identify the most significant factors.
    3. Perform follow-up experiments focusing on the significant factors for process optimization.
    """)

st.write("""
## Note on Plackett-Burman Design
The Plackett-Burman design is generated using the `pbdesign(n_factors)` function from the pyDOE2 library. 
This function creates a design matrix where each column represents a factor and each row represents an experimental run. 
The values -1 and +1 in the matrix correspond to the low and high levels of each factor, respectively.
""")
