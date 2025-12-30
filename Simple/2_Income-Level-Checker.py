import streamlit as st

st.title("Income Level Checker")

income = st.number_input("Enter your income", step=1)

if income >= 20000:
    result = """<span style="color:green;">$$$ you are SUPER HIGH INCOME.............$$ no wonder drive Ferrari$........can please PAYNOW Alvin $8888? ........Alvin is really very very poor.....</span>"""
elif income >= 15000:
    result = """<span style="color:green;">$$$ you are High income.........$$ no wonder wear Rolex$............please donate your CDC vouchers to Alvin please?...... Alvin needs to buy rice....</span>"""
elif income >= 10000:
    result = """<span style="color:orange;">$$$ you are Median High income....... $$ no wonder stay Orchard$....... can treat Alvin eat Chicken Chop Please?........YOU ARE SO RICH STILL SO STINGY???</span>"""
elif income >= 8000:
    result = """<span style="color:orange;">$$$ you are still High income........$$ can buy Alvin Starbucks or Luckin Coffee$ PLEASE? .................YOU STILL SO STINGY???</span>"""
else:
    result = """<span style="color:red;">Alvin POK KAI NO MONEY!!!..........</span>"""

# Use Markdown to render the result with HTML
st.markdown(f"${income} = {result}", unsafe_allow_html=True)
