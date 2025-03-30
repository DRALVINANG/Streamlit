#--------------------------
# Step 1: Install and Import Libraries
#--------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

#--------------------------
# Step 2: Load Data Functions
#--------------------------

def load_gender_data():
    """Load gender-based health behaviors data."""
    gender = pd.read_csv("https://raw.githubusercontent.com/tertiarycourses/datasets/master/health-behaviours-among-secondary-school-students-by-gender.csv",
                         index_col='year',
                         usecols=['year', 'gender', 'physical_activity', 'vegetable_intake', 'sweetened_drinks_intake', 'fat_intake'])
    return gender

def load_edu_data():
    """Load educational level-based health behaviors data."""
    education = pd.read_csv("https://raw.githubusercontent.com/tertiarycourses/datasets/master/health-behaviours-among-secondary-school-students-by-educational-level.csv",
                            index_col='year',
                            usecols=['year', 'edu_level', 'physical_activity', 'vegetable_intake', 'sweetened_drinks_intake', 'fat_intake'])
    return education

#--------------------------
# Step 3: Data Concatenation Function
#--------------------------

def concatenate_data(gender_data, edu_data):
    """Concatenate gender and education dataframes."""
    combined = pd.concat([gender_data, edu_data], axis=0)
    return combined

#--------------------------
# Step 4: Data Visualization Functions
#--------------------------

def plot_histogram():
    """Plot a histogram based on the provided dataset."""
    # Example dataset for histogram
    data = [50, 100, 150, 150, 200, 250, 100, 200, 300, 50, 100, 100, 300]
    plt.hist(data, bins=10)
    plt.title("Histogram of Sample Data")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    st.pyplot(plt)
    st.write("This histogram shows the frequency distribution of the sample data, which helps visualize the spread of values.")

def plot_gender_year_bar(gender_data):
    """Plot a bar chart for gender data by year."""
    gender_data.pivot_table(index='year', columns='gender', values='physical_activity').plot.bar()
    plt.title("Physical Activity by Gender over the Years")
    plt.xlabel('Year')
    plt.ylabel('Physical Activity')
    st.pyplot(plt)
    st.write("This bar chart shows the physical activity trends over the years for different genders.")

def plot_stacked_bar(gender_data, edu_data):
    """Plot stacked bar charts for gender and educational level data."""
    # Gender data plot
    gender_data.groupby('gender').sum().plot.bar(stacked=True)
    plt.xlabel('Gender')
    plt.ylabel('Total Count')
    plt.title('Health Behaviors by Gender')
    st.pyplot(plt)
    st.write("This stacked bar chart visualizes health behaviors for each gender, allowing comparisons across categories.")
    
    # Educational level data plot
    edu_data.groupby('edu_level').sum().plot.bar(stacked=True)
    plt.xlabel('Educational Level')
    plt.ylabel('Total Count')
    plt.title('Health Behaviors by Educational Level')
    st.pyplot(plt)
    st.write("This chart compares health behaviors across educational levels, highlighting trends based on education.")

def plot_pivot_table(gender_data, edu_data):
    """Plot pivot tables for physical activity by gender and education level."""
    gender_data.pivot_table(index='year', columns='gender', values='physical_activity').plot.bar()
    plt.title("Physical Activity by Gender over the Years")
    plt.xlabel('Year')
    plt.ylabel('Physical Activity')
    st.pyplot(plt)
    st.write("This pivot table chart shows the distribution of physical activity based on gender over time.")
    
    edu_data.pivot_table(index='year', columns='edu_level', values='physical_activity').plot.bar()
    plt.title("Physical Activity by Education Level over the Years")
    plt.xlabel('Year')
    plt.ylabel('Physical Activity')
    st.pyplot(plt)
    st.write("This chart compares physical activity by educational level over the years, providing insights into educational trends.")

def plot_summary_statistics(gender_data):
    """Display summary statistics for health behaviors."""
    result = gender_data[['physical_activity','vegetable_intake','sweetened_drinks_intake']].describe()
    st.write("Summary Statistics for Physical Activity, Vegetable Intake, and Sweetened Drinks Intake:")
    st.dataframe(result)
    st.write("The table above summarizes key statistics for physical activity, vegetable intake, and sweetened drinks intake.")

#--------------------------
# Step 5: Streamlit Application Setup
#--------------------------

# Streamlit UI
st.title('Health Behaviors Among Secondary School Students')

# Adding a brief description below the title to guide the user
st.write("""
This app visualizes health behaviors among secondary school students based on gender and educational level. 
The app shows trends over the years and provides summary statistics for various health behaviors.
""")

st.sidebar.header('Select Data Type')

# Load gender and education data
gender_data = load_gender_data()
edu_data = load_edu_data()

# Concatenate data
combined_data = concatenate_data(gender_data, edu_data)

# Display raw data
st.write("### Raw Data (Concatenated):")
st.dataframe(combined_data)

# Add a horizontal line after the section
st.markdown("---")

# Display visualizations
st.write("### Visualizations of Health Behaviors:")

# Histogram of data (like the first image)
plot_histogram()

# Add a horizontal line after the histogram
st.markdown("---")

# Gender vs Year Bar Chart (like the second image)
plot_gender_year_bar(gender_data)

# Add a horizontal line after the bar chart
st.markdown("---")

# Pivot Tables and Bar Charts for Physical Activity by Gender and Education Level
plot_pivot_table(gender_data, edu_data)

# Add a horizontal line after the pivot table
st.markdown("---")

# Gender and Education Level Health Behaviors Visualization
plot_stacked_bar(gender_data, edu_data)

# Add a horizontal line after the stacked bar charts
st.markdown("---")

# Summary Statistics for Gender Data
plot_summary_statistics(gender_data)

# Add a horizontal line after the summary statistics
st.markdown("---")

# Footer Message
st.write("### Created by Dr. Alvin Ang")

#--------------------------
# Run the app with: streamlit run <script_name.py>
#--------------------------


