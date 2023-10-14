import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load data (replace 'heart.xls' with your dataset)
df = pd.read_csv("heart.xls")
st.title("Heart Disease dataset Analysis")

# Allow user to select column1 and column2
column1 = st.sidebar.selectbox("Select Column 1:", df.columns)
column2 = st.sidebar.selectbox("Select Column 2:", df.columns)

# Ask user for plot type, sub-category
plot_type = st.selectbox("Choose a plot type:", ["Scatter Plot", "Line Plot", "Bar Chart", "Histogram", "Relational Plot", "Distribution Plot"])
sub_category = ""

# Slider for number of bins (enabled only for histogram plots)
num_bins = 20
if plot_type == "Histogram":
    num_bins = st.slider("Number of Bins", min_value=1, max_value=50, value=20)

# Buttons to generate plots
generate_plot = st.button("Generate Plots")

# Display plots based on user selection after button click
if generate_plot:
    if plot_type == "Scatter Plot":
        st.subheader(f"Scatter Plot for {column1} vs {column2}")
        sns.scatterplot(data=df, x=column1, y=column2, hue="target", alpha=0.7)
        plt.xlabel(column1)
        plt.ylabel(column2)
        plt.title(f"Scatter Plot of {column1} vs {column2} by diagnosis")
        st.pyplot()
    elif plot_type == "Line Plot":
        st.subheader(f"Line Plot for {column1} vs {column2}")
        sns.lineplot(data=df, x=column1, y=column2, hue="target")
        plt.xlabel(column1)
        plt.ylabel(column2)
        plt.title(f"Line Plot of {column1} vs {column2} by diagnosis")
        st.pyplot()
    elif plot_type == "Bar Chart":
        st.subheader(f"Bar Chart for {column1} grouped by {column2}")
        st.bar_chart(df.groupby(column2)[column1].value_counts().unstack().fillna(0))
        plt.xlabel(column2)
        plt.title(f"Bar Chart of {column1} grouped by {column2}")
    elif plot_type == "Histogram":
        st.subheader(f"Histogram for {column1}")
        sns.histplot(data=df, x=column1, hue="target", bins=num_bins, kde=True)
        plt.xlabel(column1)
        plt.title(f"Histogram of {column1} by target")
        st.pyplot()
        st.subheader(f"Histogram for {column2}")
        sns.histplot(data=df, x=column2, hue="target", bins=num_bins, kde=True)
        plt.xlabel(column2)
        plt.title(f"Histogram of {column2} by target")
        st.pyplot()
    elif plot_type == "Relational Plot":
        st.subheader(f"Relational Plot for {column1} vs {column2}")
        sns.relplot(data=df, x=column1, y=column2, hue="target", kind="scatter")
        plt.xlabel(column1)
        plt.ylabel(column2)
        plt.title(f"Relational Plot of {column1} vs {column2} by target")
        st.pyplot()
    elif plot_type == "Distribution Plot":
        st.subheader(f"Distribution Plot for {column1}")
        sns.displot(data=df, x=column1, hue="target", kind="kde", fill=True)
        plt.xlabel(column1)
        plt.title(f"Distribution Plot of {column1} by target")
        st.pyplot()
        st.subheader(f"Distribution Plot for {column2}")
        sns.displot(data=df, x=column2, hue="target", kind="kde", fill=True)
        plt.xlabel(column2)
        plt.title(f"Distribution Plot of {column2} by target")
        st.pyplot()
