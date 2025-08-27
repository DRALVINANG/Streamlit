import streamlit as st

def grocery(order):
    discount = 25 if order > 200 else 0
    disc_amt = discount * order / 100
    tax = 0.07 * (order - disc_amt)
    return disc_amt, tax

st.title("Grocery Discount Calculator")

order = st.number_input("Enter the amount of your order", step=0.01)

if order > 0:
    discount, tax = grocery(order)
    st.write(f"The discount is ${discount:.2f}")
    st.write(f"The tax is ${tax:.2f}")
else:
    st.write("Please enter a valid order amount.")
