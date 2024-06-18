import streamlit as st
import numpy as np
from pyDOE2 import fullfact
import pandas as pd

st.title("Dr. Alvin's Full-Factorial Design App 1")

# Define the levels for each factor
temperatures = [20, 30]
pressures = [100, 200, 300]

# Create a list of the number of levels for each factor
levels = [len(temperatures), len(pressures)]

# Create the full-factorial design
design_matrix = fullfact(levels)

# Convert the design matrix to a more readable format
experiment_conditions = []
for row in design_matrix:
    temperature = temperatures[int(row[0])]
    pressure = pressures[int(row[1])]
    experiment_conditions.append([temperature, pressure])

# Create a Pandas DataFrame with appropriate headers
df = pd.DataFrame(experiment_conditions, columns=['Temperature (°C)', 'Pressure (psi)'])

# Display the experiment conditions in a table
st.write("In this example, we have two factors: temperature and pressure.")
st.write("The full-factorial design includes all possible combinations of these two factors.")
st.write("Temperature: 20°C & 30°C")
st.write("Pressure: 100, 200 & 300 PSI")
st.table(df)

# Create a download button for the CSV file
csv = df.to_csv(index=False)
st.download_button("Download CSV", csv, "experiment_conditions.csv", "text/csv")
