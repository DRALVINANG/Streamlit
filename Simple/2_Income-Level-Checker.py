import streamlit as st

st.title("Income Level Checker")

income = st.number_input("Enter your income", step=1)

if income >= 20000:
    result = f"""<span style="color:red;">$${income}$$ = $$$ you are SUPER HIGH INCOME</span>.............$$ no wonder drive Ferrari $$........can please PAYNOW Alvin $8888? ........<span style="color:yellow;">Alvin is really very very poor.....</span>"""
elif income >= 15000:
    result = f"""<span style="color:green;">$${income}$$ = $$$ you are High income.........$$ no wonder wear Rolex $$............please donate your CDC vouchers to Alvin please?...... Alvin needs to buy rice....</span>"""
elif income >= 10000:
    result = f"""<span style="color:orange;">$${income}$$ = $$$ you are Median High income....... $$ no wonder stay Orchard $$....... can treat Alvin eat Chicken Chop Please?........YOU ARE SO RICH STILL SO STINGY???</span>"""
elif income >= 8000:
    result = f"""<span style="color:yellow;">$${income}$$ = $$$ you are still High income........$$ can buy Alvin Starbucks or Luckin Coffee $$ PLEASE? .................YOU STILL SO STINGY???</span>"""
else:
    result = f"""<span style="color:red;">${income} = Alvin POK KAI NO MONEY!!!..........</span>"""

# Now render the complete result
st.markdown(result, unsafe_allow_html=True)
