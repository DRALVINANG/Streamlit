import streamlit as st
import numpy as np
import pandas as pd
from pyDOE2 import fullfact

# Create a Streamlit app
st.title("Dr. Alvin's Full-Factorial Design App 2")
st.subheader("One Factor At a Time (OFAT)")

# Get the number of factors from the user
num_factors = st.number_input("Enter the number of factors:", min_value=1, value=2)

# Get the levels for each factor from the user
factors = []
for i in range(num_factors):
    st.markdown("***")  # Add a line separator
    factor_name = st.text_input(f"**Enter factor {i+1} name:**", value=f"Factor {i+1}")
    levels = st.number_input(f"Enter the number of levels for {factor_name}: ", min_value=2, value=2)
    factor_levels = []
    for j in range(levels):
        level_value = st.number_input(f"Enter level {j+1} value for {factor_name}: ", value=j+1)
        factor_levels.append(level_value)
    factors.append((factor_name, factor_levels))

# Create the full-factorial design
levels = [len(factor[1]) for factor in factors]
design_matrix = fullfact(levels)

# Convert the design matrix to a more readable format
experiment_conditions = []
for row in design_matrix:
    condition = []
    for i, factor in enumerate(factors):
        condition.append(factor[1][int(row[i])])
    experiment_conditions.append(condition)

st.markdown("***")  # Add a line separator
st.write("**Experiment Conditions:**")  # Bold the header
st.table(pd.DataFrame(experiment_conditions, columns=[factor[0] for factor in factors]))

# Add a button to download the results as a CSV file
@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(pd.DataFrame(experiment_conditions, columns=[factor[0] for factor in factors]))

st.download_button(
    label="Download as CSV",
    data=csv,
    file_name='experiment_conditions.csv',
    mime='text/csv'
)
