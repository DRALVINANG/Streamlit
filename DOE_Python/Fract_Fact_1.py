import streamlit as st
import pandas as pd
from pyDOE2 import fracfact
import numpy as np

st.title("Fractional Factorial Design for Coffee Brewing Optimization")

st.write("This app demonstrates a 2^(5-2) fractional factorial design for optimizing coffee brewing parameters.")
st.write("The factors are 'A B C AB AC'")
st.write("The fractional factorial design is created using the code 'design = fracfact(factors)'")
st.write("There are 5 factors: ")
st.write("    'A': 'Grind Size', ['Fine', 'Coarse']")
st.write("    'B': 'Water Temperature', ['90°C', '96°C']")
st.write("    'C': 'Brewing Time', ['3 min', '5 min']")
st.write("    'D': 'Coffee-to-Water Ratio', ['1:15', '1:17']")
st.write("    'E': 'Roast Level', ['Light', 'Dark']")

# Define the factors for a 2^(5-2) fractional factorial design
factors = 'A B C AB AC'

# Generate the fractional factorial design
design = fracfact(factors)

# Define factor names and levels
factor_names = {
    'A': 'Grind Size',
    'B': 'Water Temperature',
    'C': 'Brewing Time',
    'D': 'Coffee-to-Water Ratio',
    'E': 'Roast Level'
}

factor_levels = {
    'A': ['Fine', 'Coarse'],
    'B': ['90°C', '96°C'],
    'C': ['3 min', '5 min'],
    'D': ['1:15', '1:17'],
    'E': ['Light', 'Dark']
}

# Convert design to DataFrame
df = pd.DataFrame(design, columns=['A', 'B', 'C', 'D', 'E'])

# Replace -1 and 1 with actual factor levels
for col in df.columns:
    df[col] = df[col].map({-1: factor_levels[col][0], 1: factor_levels[col][1]})

# Rename columns to descriptive names
df.columns = [factor_names[col] for col in df.columns]

st.subheader("Experimental Design")
st.dataframe(df)

st.subheader("Design Statistics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total number of runs", len(df))
    st.metric("Full factorial runs", 2**5)
with col2:
    st.metric("Reduction in runs", 2**5 - len(df))
    st.metric("Fraction of full factorial", f"1/{2**(5-3)}")

st.subheader("Explanation of Factor Selection")
st.write("We chose A, B, and C as our base factors for the following reasons:")
reasons = [
    "These are likely the most influential factors in coffee brewing.",
    "Using a 2^(5-2) design allows us to study all 5 factors with fewer runs.",
    "Factors D and E are generated from interactions of A, B, and C.",
    "This design balances efficiency with the ability to study main effects and some interactions."
]
for i, reason in enumerate(reasons, 1):
    st.write(f"{i}. {reason}")

st.subheader("Design Details")
st.write("- This is a Resolution III design.")
st.write("- Main effects are confounded with two-factor interactions.")
st.write("- Allows estimation of main effects with minimal runs.")
