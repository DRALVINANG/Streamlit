import streamlit as st

st.title("Income Level Checker")

income = st.number_input("Enter your income", step=1)

if income >= 20000:
    result = "you are SUPER HIGH INCOME.............no wonder drive Ferrari........can please PAYNOW Alvin $88888?........ Alvin is really very very poor....."
elif income >= 15000:
    result = "you are High income.........no wonder wear Rolex............please donate your CDC vouchers to Alvin please?...... Alvin needs to buy rice...."
elif income >= 10000:
    result = "you are Median High income....... no wonder stay Orchard....... can treat Alvin eat Chicken Chop Please?........YOU ARE SO RICH STILL SO STINGY???"
elif income >= 8000:
    result = "you are still High income........can buy Alvin Starbucks or Luckin Coffee PLEASE? .................YOU STILL SO STINGY???"
else:
    result = "Alvin POK KAI NO MONEY!!!.........."

st.write(f"{income} = {result}")
