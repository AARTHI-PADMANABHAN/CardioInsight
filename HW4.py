import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df_heart_disease = pd.read_csv("heart.xls")
pd.set_option('mode.use_inf_as_null', True)

# Set a white background for Seaborn plots
sns.set(style="whitegrid")
sns.set_context("talk", rc={"axes.facecolor": "white"})

# KDE Plot
st.title("Heart Disease Age Distribution (KDE)")
sns.displot(data=df_heart_disease, x="age", hue="target", kind="kde")
st.pyplot()

# Scatter Plot
st.title("Heart Disease Cholesterol vs. FBS (Scatter Plot)")
sns.scatterplot(data=df_heart_disease, x="chol", y="fbs", hue="target", alpha=0.7)
st.pyplot()

# Regression Plot
st.title("Regression Plot: Cholesterol vs. Target")
sns.regplot(x="chol", y="target", data=df_heart_disease)
st.pyplot()
