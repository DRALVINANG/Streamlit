import streamlit as st

# CPF Calculation Function
def cpf(age, wage):
    if age < 16:
        cpfa = 0
        cpfb = 0
    elif wage < 750:
        cpfa = 0
        cpfb = 0
    elif age <= 55:
        cpfa = 0.17
        cpfb = 0.2
    elif age <= 60:
        cpfa = 0.13
        cpfb = 0.13
    elif age <= 65:
        cpfa = 0.09
        cpfb = 0.075
    else:
        cpfa = 0.075
        cpfb = 0.05

    cpfX = wage * cpfa
    cpfY = wage * cpfb
    return cpfX, cpfY

# Streamlit App Layout
st.title("CPF Contribution Calculator")
st.markdown("""
    This application calculates the **Central Provident Fund (CPF)** contributions 
    for both employer and employee based on the individual's age and monthly wage. 
    Simply enter your details below to find out how much is contributed towards your CPF!
""")

# User Input
age = st.number_input("Enter your Age:", min_value=0, max_value=100, value=25)
wage = st.number_input("Enter your Monthly Wage ($):", min_value=0, value=1000)

# Calculate Contributions
if st.button("Calculate CPF Contributions"):
    cpfX, cpfY = cpf(age, wage)
    st.success(f"Employer Contribution: **${cpfX:0.2f}**")
    st.success(f"Employee Contribution: **${cpfY:0.2f}**")

# Footer
st.markdown("---")
st.markdown("© 2024 CPF Contribution Calculator. All rights reserved.")
