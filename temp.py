# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/aarthipadmanabhan/Desktop/data.csv")
st.title("WI Cancer datset Analysis")
chosen_column = st.selectbox("Choose a column:", df.columns)
if chosen_column:
    st.subheader(f"seaborn plot for {chosen_column}")
    sns.histplot(df, x=chosen_column, hue="diagnosis", kde=True)
    plt.xlabel(chosen_column)
    plt.title(f"Plot of {chosen_column} by diagnosis")
    st.pyplot()
    st.write("Basic Information of data:")
st.write(df.describe())
st.write("Distribution Plot:")
st.bar_chart(df["diagnosis"].value_counts())
