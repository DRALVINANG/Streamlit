import streamlit as st
import numpy as np
from pyDOE2 import bbdesign
import pandas as pd

st.title("Customizable Box-Behnken Experiment Design")

st.write("""
## Box-Behnken Design (BBD) Overview

- Minimum 3 Factors (Maximum 10 Factors)
- No Corner Points = All Center Points

BB design simply gives us a better idea of how the Response moves with ALL the Factors, but doesn't tell us which Factors need to be removed.

### Why do we need BBD?

**Reason 1: Helps to Avoid Extreme Conditions**

Since BBD are all Center points, it avoids extreme experimental conditions: such as Corner points and Star points. Extreme conditions can sometimes be impractical or even dangerous to create.

**Reason 2: Helps to Troubleshoot or Optimize a Process**

BBD is specifically designed to fit a second-order model. It is able to capture both Linear and Quadratic effects of the independent variables, as well as the interaction effects more clearly (compared to the CCD).

Thus, an Engineer may use BBD to identify the factors that are leading to defects in a product.

**Reason 3: Efficiency in Number of Runs**

Box-Behnken designs are much more efficient in terms of the number of required runs compared to full factorial designs, especially as the number of factors increases.
""")

# Let user choose the number of factors
num_factors = st.slider("Select the number of factors:", min_value=3, max_value=10, value=3)

# Calculate number of runs for full factorial and Box-Behnken
full_factorial_runs = 3**num_factors
bb_runs = len(bbdesign(num_factors))

st.write(f"""
### Comparison of Number of Runs:

- Full Factorial Design (3 levels): {full_factorial_runs} runs
- Box-Behnken Design: {bb_runs} runs

**Reduction in runs: {full_factorial_runs - bb_runs} ({((full_factorial_runs - bb_runs) / full_factorial_runs * 100):.1f}%)**

The Box-Behnken design significantly reduces the number of required experimental runs while still capturing the main effects and interactions.
""")

# Create input fields for factor names and levels
factors = []
factor_levels = {}

for i in range(num_factors):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        factor_name = st.text_input(f"Factor {i+1} name:", value=f"Factor {i+1}")
    
    with col2:
        level_low = st.text_input(f"Factor {i+1} low level (-1):", value="Low")
    
    with col3:
        level_mid = st.text_input(f"Factor {i+1} mid level (0):", value="Medium")
    
    with col4:
        level_high = st.text_input(f"Factor {i+1} high level (1):", value="High")
    
    factors.append(factor_name)
    factor_levels[factor_name] = {-1: level_low, 0: level_mid, 1: level_high}

# Generate the Box-Behnken design
design = bbdesign(num_factors)

# Create a DataFrame
df = pd.DataFrame(design, columns=factors)

# Replace -1, 0, 1 with actual factor levels
for factor in factors:
    df[factor] = df[factor].map(factor_levels[factor])

# Add a run order column
df.insert(0, "Run Order", range(1, len(df) + 1))

st.write("Experiment Design Plan:")
st.dataframe(df)

st.write(f"This design includes {len(df)} experimental runs, including center points for assessing experimental error and checking for curvature in the response.")

st.write("Instructions:")
st.write("1. Conduct each experimental run in the order specified.")
st.write("2. For each run, set up your experiment with the specified factor levels.")
st.write("3. Measure and record the response variable(s) for each run.")
st.write("4. After completing all runs, analyze the data to determine the optimal combination of factors.")

# Add an option to download the design as a CSV file
csv = df.to_csv(index=False)
st.download_button(
    label="Download experiment design as CSV",
    data=csv,
    file_name="box_behnken_design.csv",
    mime="text/csv",
)
