import streamlit as st

# --------------------------------------------------------------------------------------------------
def time_value_money(PV, interest_rate, period):
    FV = PV * (1 + interest_rate) ** period
    return FV

#--------------------------------------------------------------------------------------------------
st.title("Time Value of Money Calculator")

## Input section
st.header("Input Parameters")

PV = st.number_input("Present Value (PV)", value=1000.0, step=100.0)

interest_rate = st.number_input("Interest Rate (decimal)", 
                                value=0.05, 
                                step=0.01, 
                                format="%.2f")

period = st.number_input("Period (years)", 
                         value=5, 
                         step=1)

## Calculation
if st.button("Calculate Future Value"):
    FV = time_value_money(PV, interest_rate, period)
    st.success(f"The Future Value is: ${FV:.2f}")

#--------------------------------------------------------------------------------------------------
st.markdown("""
### About
This app calculates the future value of an investment based on the present value, interest rate, and time period using the time value of money formula:

FV = PV * (1 + r)^t

Where:
- FV is the future value
- PV is the present value
- r is the interest rate
- t is the time period (in years)
""")
