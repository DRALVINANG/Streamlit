import streamlit as st

st.title("Income Level Checker")

income = st.number_input("Enter your income", step=1)

if income >= 20000:
    result = "Super High income - no wonder Ferrari Car"
elif income >= 15000:
    result = "High income - no wonder Rolex Watch"
elif income >= 10000:
    result = "Median income - no wonder stay Orchard"
elif income >= 8000:
    result = "Normal income - where got low??"
else:
    result = "Alvin POK KAI NO MONEY!!!"

st.write(f"{income} = {result}")
