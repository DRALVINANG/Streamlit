import streamlit as st

# Add a title
st.title("My First Streamlit App")

# Add some text
st.write("Hello, World!")

# Add an input widget
name = st.text_input("Enter your name", "Type here...")
if name:
    st.write(f"Hello, {name}!")
