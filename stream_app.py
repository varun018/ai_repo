import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Simple Data Dashboard",
    page_icon="📊",
    layout="wide"
)

# Main title
st.title("📊 Simple Data Dashboard")
st.markdown("Welcome to this interactive Streamlit app!")

# Sidebar
st.sidebar.header("Settings")
st.sidebar.markdown("Customize your dashboard here:")

# Sidebar controls
num_points = st.sidebar.slider("Number of data points", 10, 1000, 100)
chart_type = st.sidebar.selectbox("Chart Type", ["Line", "Bar", "Scatter"])
show_raw_data = st.sidebar.checkbox("Show raw data", False)

# Generate sample data
@st.cache_data
def generate_data(n_points):
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq='D')
    np.random.seed(42)  # For reproducible results
    values = np.cumsum(np.random.randn(n_points)) + 100
    categories = np.random.choice(['Category A', 'Category B', 'Category C'], n_points)
    
    df = pd.DataFrame({
        'Date': dates,
        'Value': values,
        'Category': categories,
        'Random': np.random.randn(n_points) * 10 + 50
    })
    return df

# Generate the data
data = generate_data(num_points)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Data Visualization")
    
    # Create chart based on selection
    if chart_type == "Line":
        fig = px.line(data, x='Date', y='Value', title="Value Over Time")
    elif chart_type == "Bar":
        fig = px.bar(data.groupby('Category')['Value'].mean().reset_index(), 
                     x='Category', y='Value', title="Average Value by Category")
    else:  # Scatter
        fig = px.scatter(data, x='Value', y='Random', color='Category', 
                        title="Value vs Random (by Category)")
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📋 Summary Statistics")
    
    # Display key metrics
    col2_1, col2_2 = st.columns(2)
    
    with col2_1:
        st.metric("Total Points", len(data))
        st.metric("Average Value", f"{data['Value'].mean():.2f}")
    
    with col2_2:
        st.metric("Max Value", f"{data['Value'].max():.2f}")
        st.metric("Min Value", f"{data['Value'].min():.2f}")
    
    # Category breakdown
    st.subheader("Category Distribution")
    category_counts = data['Category'].value_counts()
    st.bar_chart(category_counts)

# Raw data section
if show_raw_data:
    st.subheader("🔍 Raw Data")
    st.dataframe(data, use_container_width=True)

# Interactive input section
st.subheader("💬 Interactive Section")

col3, col4 = st.columns(2)

with col3:
    user_name = st.text_input("Enter your name:", "")
    user_age = st.number_input("Enter your age:", min_value=1, max_value=120, value=25)

with col4:
    favorite_color = st.selectbox("Favorite color:", ["Red", "Blue", "Green", "Yellow", "Purple"])
    rating = st.slider("Rate this app (1-10):", 1, 10, 5)

if user_name:
    st.success(f"Hello {user_name}! You're {user_age} years old and your favorite color is {favorite_color}. Thanks for rating this app {rating}/10!")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")