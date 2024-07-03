import streamlit as st
import numpy as np
import pandas as pd
from pyDOE2 import pbdesign

# Define the number of factors
n_factors = 16

# Generate the Plackett-Burman design
design = pbdesign(n_factors)

# Create a Streamlit app
st.title("Plackett-Burman Design for Plastic Bottle Manufacturing Optimization")

st.write("""
## Problem Statement
A plastic bottle manufacturing company is experiencing inconsistent quality in their production process. 
They want to identify the most significant factors affecting bottle quality out of 16 potential factors. 
The goal is to optimize the manufacturing process to improve bottle quality and reduce defects.

## Why Plackett-Burman Design?
Plackett-Burman designs are two-level designs.
With 16 factors, a full factorial design would require 2^16 = 65,536 runs, which is impractical. 
Plackett-Burman design allows us to screen these factors with a much smaller number of runs, 
identifying the most important factors for further investigation.
\nThe code used here is : pbdesign(n_factors), where n_factors represent the number of factors.

## Factors and Levels
""")

factors = [
    "Resin Type", "Melt Temperature", "Injection Speed", "Cooling Time",
    "Mold Temperature", "Holding Pressure", "Screw Speed", "Barrel Temperature",
    "Nozzle Temperature", "Cycle Time", "Mold Closing Speed", "Ejection Speed",
    "Colorant Percentage", "Regrind Percentage", "Humidity", "Ambient Temperature"
]

levels = {
    "Resin Type": ("PET", "HDPE"),
    "Melt Temperature": ("Low: 220°C", "High: 280°C"),
    "Injection Speed": ("Slow: 50 mm/s", "Fast: 100 mm/s"),
    "Cooling Time": ("Short: 10s", "Long: 20s"),
    "Mold Temperature": ("Cool: 10°C", "Warm: 30°C"),
    "Holding Pressure": ("Low: 40 MPa", "High: 60 MPa"),
    "Screw Speed": ("Slow: 50 rpm", "Fast: 100 rpm"),
    "Barrel Temperature": ("Low: 200°C", "High: 240°C"),
    "Nozzle Temperature": ("Low: 220°C", "High: 260°C"),
    "Cycle Time": ("Short: 15s", "Long: 25s"),
    "Mold Closing Speed": ("Slow: 100 mm/s", "Fast: 200 mm/s"),
    "Ejection Speed": ("Slow: 100 mm/s", "Fast: 200 mm/s"),
    "Colorant Percentage": ("Low: 0.5%", "High: 2%"),
    "Regrind Percentage": ("Low: 10%", "High: 30%"),
    "Humidity": ("Low: 30%", "High: 70%"),
    "Ambient Temperature": ("Cool: 20°C", "Warm: 30°C")
}

for factor, (low, high) in levels.items():
    st.write(f"- **{factor}**: {low} (-1), {high} (+1)")

st.write("## Plackett-Burman Design Matrix")
design_df = pd.DataFrame(design, columns=factors)
st.dataframe(design_df)

st.write(f"Number of runs in Plackett-Burman design: {len(design)}")
st.write(f"Number of runs in full factorial design: {2**n_factors}")
st.write(f"Number of runs saved: {2**n_factors - len(design)}")

st.write("""
## Next Steps
1. Conduct the experiments according to this design.
2. Analyze the results to identify the most significant factors.
3. Perform follow-up experiments focusing on the significant factors for process optimization.
""")
