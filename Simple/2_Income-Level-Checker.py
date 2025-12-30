import streamlit as st

st.title("Income Level Checker")

income = st.number_input("Enter your income", step=1)

if income >= 20000:
    result = "SUPER HIGH INCOME -> no wonder drive Ferrari -> KINDLY PAYNOW Alvin $88888 THANK YOU SO MUCH! Alvin is really very very poor..SO RICH STILL SO STINGY???"
elif income >= 15000:
    result = "High income -> no wonder wear Rolex -> PLEASE DONATE ALVIN YOUR CDC VOUCHERS THANK YOU... Alvin needs to buy rice....SO RICH STILL SO STINGY???"
elif income >= 10000:
    result = "Median High income -> no wonder stay Orchard -> can treat Alvin eat Chicken Chop Please? SO RICH STILL SO STINGY???"
elif income >= 8000:
    result = "Also High income -> can buy Alvin Starbucks or Luckin Coffee PLEASE? SO RICH STILL SO STINGY???"
else:
    result = "Alvin POK KAI NO MONEY!!!"

st.write(f"{income} = {result}")
