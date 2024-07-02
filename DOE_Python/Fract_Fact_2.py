import streamlit as st
import pandas as pd
from pyDOE2 import fracfact
import numpy as np
import base64

st.title("Customizable Fractional Factorial Design Generator")

st.write("This app allows you to create a customized fractional factorial design with up to 9 factors.")

# User inputs
num_factors = st.slider("Number of factors (2-9):", min_value=2, max_value=9, value=3)

factor_names = {}
factor_levels = {}

for i in range(num_factors):
    col1, col2, col3 = st.columns(3)
    with col1:
        factor = chr(65 + i)  # A, B, C, ..., I
        name = st.text_input(f"Name for factor {factor}:", value=f"Factor {factor}")
        factor_names[factor] = name
    with col2:
        level1 = st.text_input(f"Level 1 for factor {factor}:", value="Low")
    with col3:
        level2 = st.text_input(f"Level 2 for factor {factor}:", value="High")
    factor_levels[factor] = [level1, level2]

# Generate design string
def generate_design_string(n):
    if n <= 4:
        return ' '.join(chr(65+i) for i in range(n))
    elif n <= 7:
        base = ' '.join(chr(65+i) for i in range(4))
        generators = 'AB AC BC'
        return f"{base} {generators[:3*(n-4)]}"
    else:  # 8 or 9 factors
        base = ' '.join(chr(65+i) for i in range(5))
        generators = 'ABC ABD ACD BCD'
        return f"{base} {generators[:4*(n-5)]}"

design_string = generate_design_string(num_factors)

st.write(f"Design string: {design_string}")

# Generate the fractional factorial design
design = fracfact(design_string)

# Convert design to DataFrame
df = pd.DataFrame(design, columns=[chr(65+i) for i in range(num_factors)])

# Replace -1 and 1 with actual factor levels
for col in df.columns:
    df[col] = df[col].map({-1: factor_levels[col][0], 1: factor_levels[col][1]})

# Rename columns to descriptive names
df.columns = [factor_names[col] for col in df.columns]

st.subheader("Experimental Design")
st.dataframe(df)

# Download link
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="fractional_factorial_design.csv">Download CSV File</a>'
    return href

st.markdown(get_table_download_link(df), unsafe_allow_html=True)

st.subheader("Design Statistics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total number of runs", len(df))
    st.metric("Full factorial runs", 2**num_factors)
with col2:
    st.metric("Reduction in runs", 2**num_factors - len(df))
    st.metric("Fraction of full factorial", f"1/{2**(num_factors-len(df).bit_length())}")

st.subheader("Design Details")
st.write(f"- This is a 2^({num_factors}-{num_factors-len(df).bit_length()}) fractional factorial design.")
st.write("- Main effects are estimated, but may be confounded with higher-order interactions.")
st.write("- As the number of factors increases, the design becomes more fractionated.")
st.write("- This design balances efficiency with the ability to study main effects.")
st.write("- For designs with many factors, careful consideration of aliasing is necessary.")

# Resolution calculation
resolution = min(len(design_string.split()), 3)
st.write(f"- This design has at least Resolution {resolution}.")
