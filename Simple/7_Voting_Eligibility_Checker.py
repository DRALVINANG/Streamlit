import streamlit as st

# Title and description of the app
st.title("Singapore Voting Eligibility Checker")
st.markdown("""
    Welcome to the Voting Eligibility Checker! This application determines if you are eligible to vote in Singapore based on your age.
    
    Please enter your age below. Feel free to test the app by entering different types of inputs, including invalid ones (like letters or symbols) to see how it handles errors!
""")

# Input for age
age_input = st.text_input("What is your age?")  # Prompt user for their age

# Button to check eligibility
if st.button("Check Eligibility"):
    if age_input:  # Check if the input is not empty
        try:
            age = int(age_input)  # Attempt to convert input to an integer
            
            if age < 21:
                st.warning("Minimum age for voting is 21. Please try again.")
            else:
                st.success(f"You are {age} years old. You are eligible to vote in Singapore.")
                
        except ValueError:
            st.error("Invalid input. Please enter a valid age (integer).")
    else:
        st.info("Please enter your age to check your voting eligibility.")
