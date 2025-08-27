import streamlit as st

st.title("Income Level Checker")

income = st.number_input("Enter your income", step=1)

if income > 8000:
    result = "High income"
elif income > 4000:
    result = "Medium high income"
elif income > 2000:
    result = "Medium income"
elif income > 1000:
    result = "Medium low income"
else:
    result = "Low income"

st.write(f"Based on your income of {income}, your income level is: {result}")
