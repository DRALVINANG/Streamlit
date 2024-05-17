import streamlit as st

def fare(d):
    book = 2.0
    start = 3.0
    cost = 1.0
    fare = book + start + d * cost
    return fare

st.title("Fare Calculator")

## Dropdown List
distance_options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
selected_distance = st.selectbox("Select the distance (in km)", distance_options)

## Calculate Fare
calculated_fare = fare(selected_distance)

## Display Result
st.write(f"The fare for a distance of {selected_distance} km is ${calculated_fare:.2f}")
