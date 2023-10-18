
import pandas as pd
import streamlit as st
import seaborn as sns
import altair as alt

st.set_option('deprecation.showPyplotGlobalUse', False)

# Center-aligned title
st.markdown("<h1 style='text-align: center;'>Heart Disease Data Analysis</h1>", unsafe_allow_html=True)
st.write("CMSE Mid-Term Project, by Aarthi Padmanabhan")

# Image with caption
st.image("Heart_image.png", caption="Heart Disease Analysis", use_column_width=True)

# Project description
description_lines = [
    "In the realm of healthcare and cardiology, understanding the factors influencing heart disease is crucial for prevention and effective treatment.",
    "Various aspects, such as age, gender, chest pain type, resting blood pressure, cholesterol levels, fasting blood sugar, electrocardiographic measurements, maximum heart rate achieved, exercise-induced angina, and several other attributes, play pivotal roles.",
    "Analyzing the extensive dataset on heart disease allows us to explore intricate patterns and relationships among these factors.",
    "This project aims to delve into the depths of this data, seeking correlations and trends that might illuminate the pathways to heart disease.",
    "By visualizing these intricate relationships, we hope to uncover valuable insights that can aid in the early diagnosis and management of heart-related conditions."
]

# Join the lines into a single string and display
description = "\n".join(description_lines)
st.write(description)

df = pd.read_csv("heart.xls")

st.subheader("Heart Disease Data")

# Dataframe to display when the button is clicked
if st.button("View Data"):
    st.write(df)
    
st.subheader("Attribute Descriptions")

if st.button("View Attributes information"):
   st.write("**Age:** Age of the patient in years.")
   st.write("**Sex:** Gender of the patient (1 = male, 0 = female).")
   st.write("**Chest Pain Type (cp):** Type of chest pain experienced by the patient.")
   st.write("   - 3: Typical Angina - Chest pain related to decreased blood supply to the heart, relieved by rest or nitroglycerin.")
   st.write("   - 2: Non-Anginal Pain - Chest pain not related to angina.")
   st.write("   - 1: Atypical Angina - Chest pain not typical angina, related to decreased blood supply to the heart.")
   st.write("   - 0: Asymptomatic - Individuals who do not experience chest pain.")
   st.write("**Resting Blood Pressure (trestbps):** Resting blood pressure in mm Hg on admission to the hospital.")
   st.write("**Serum Cholestoral (chol):** Serum cholesterol in mg/dl.")
   st.write("**Fasting Blood Sugar (fbs):** Fasting blood sugar level > 120 mg/dl (1 = true, 0 = false).")
   st.write("**Resting Electrocardiographic Results (restecg):** Resting electrocardiographic results.")
   st.write("   - 0: Normal - Normal results.")
   st.write("   - 1: ST-T Wave Abnormality - Abnormality related to ST-T wave.")
   st.write("   - 2: Left Ventricular Hypertrophy - Showing probable or definite left ventricular hypertrophy.")
   st.write("**Maximum Heart Rate Achieved (thalach):** The person's maximum heart rate achieved.")
   st.write("**Exercise Induced Angina (exang):** Exercise-induced angina (1 = yes, 0 = no).")
   st.write("**ST Depression Induced by Exercise Relative to Rest (oldpeak):** ST depression induced by exercise relative to rest.")
   st.write("**Slope of the Peak Exercise ST Segment (slope):** The slope of the peak exercise ST segment.")
   st.write("   - 0: Upsloping - Sloping upwards indicates a better prognosis.")
   st.write("   - 1: Flat - Flat indicates no upsloping or downsloping of the ST segment.")
   st.write("   - 2: Downsloping - Sloping downwards indicates an unhealthy heart.")
   st.write("**Number of Major Vessels (ca):** Number of major vessels colored by fluoroscopy (0-3).")
   st.write("**Thalassemia (thal):** A blood disorder called thalassemia.")
   st.write("   - 1: Normal - Normal thalassemia.")
   st.write("   - 2: Fixed Defect - Fixed defect thalassemia.")
   st.write("   - 3: Reversible Defect - Reversible defect thalassemia.")

st.sidebar.markdown("_**App Objectives**_")
st.sidebar.write("This app aims to explore the Cleveland Heart Disease dataset and answer the following questions:")

# List of questions in italics
st.sidebar.markdown("""
- _What factors contribute to heart disease?_
- _Is there a correlation between certain attributes and heart disease?_
""")

st.sidebar.title("Data Visualization Options")

if st.sidebar.button("Show Summary Statistics"):
    st.subheader("Summary Statistics")
    st.write(df.describe())
    
if st.sidebar.button("Show Correlation matrix"):
    st.subheader("Summary Statistics")
    st.write(df.corr())
    
st.sidebar.subheader("Visualize distributions of data")
selected_feature = st.sidebar.selectbox("Select Feature:", df.columns)

# Visualization based on user's selection
if st.sidebar.button("View distribution"):
    st.subheader(f"Distribution of {selected_feature} by Target")
    chart = alt.Chart(df).mark_bar().encode(
    x=alt.X(selected_feature, bin=alt.Bin(maxbins=30), title=selected_feature),
    y='count():Q',
    color='target:N'
    ).properties(width=500, height=300)
    st.altair_chart(chart, use_container_width=True)

st.sidebar.subheader("Visualize statistical relationships")

visualization_type = st.sidebar.selectbox("Select Visualization Type", ["Correlation", "Scatter Plots"])

#selected_columns = st.sidebar.multiselect('Select Columns:', df.columns, ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs','restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'])

#filtered_df = df[selected_columns]
    
if visualization_type == "Correlation":
    if st.sidebar.button("View relation"):
        st.subheader('Correlation Plot')
        colormap = sns.color_palette("Greens")
        sns.heatmap(df.corr(), cmap= colormap, annot=df.corr().rank(axis="columns"))
        st.pyplot()
elif visualization_type == "View relation":
    x_column = st.sidebar.selectbox("Select X-axis Column:", df.columns)
    y_column = st.sidebar.selectbox("Select Y-axis Column:", df.columns)
    if st.sidebar.button("Generate Plot"):
        st.subheader('Scatter Plot')
        sns.relplot(data=df, x=x_column, y=y_column, hue='target', height=6)
        st.pyplot()
        
st.sidebar.subheader("Visualize multi-plot grids")

x_column = st.sidebar.selectbox("Select X-axis Column:", df.columns)
y_column = st.sidebar.selectbox("Select Y-axis Column:", df.columns)
if st.sidebar.button("View grids"):
   g = sns.FacetGrid(df, col="sex", hue="target")
   g.map(sns.scatterplot, x_column, y_column, alpha=.7)
   g.add_legend()
   st.subheader("Multi Grid Plots")
   st.pyplot()
        
st.sidebar.title("Insights and Conclusion")

if st.sidebar.button("Display information"):
    sns.displot(df, x="age", hue="target", kind="kde")
    st.pyplot()
    st.write("The distinct peaks and variations in the plot reveal a compelling pattern: there exists a specific age group within our dataset where the probability of experiencing heart disease is markedly elevated. Upon a closer analysis of the KDE plot depicting the relationship between age and the likelihood of heart disease, it appears that there is no discernible threshold or significant correlation between age and the target variable. Other factors such as chest pain type, the person's maximum heart rate achieved, and genetic predisposition could potentially play more significant roles in influencing heart health outcomes.")
    st.write(df.corr())
    st.write("From the above matrix, it is clear that chest pain and thalach(maximum heart rate) are highly correlated with the target variable when compared to the rest of the variables.")
    sns.catplot(data=df, y="cp", hue="target", kind="count", palette="pastel", edgecolor=".6",)
    st.pyplot()
    st.write("The categorical plot examining different chest pain types (cp) in relation to the presence or absence of heart disease (target) reveals a compelling insight. Specifically, individuals with chest pain type 2 (Non-anginal pain) appears to be more susceptible to heart disease, as they are the predominant group affected within this dataset. The higher count of heart disease cases among individuals experiencing this specific chest pain type suggests a potential correlation between non-anginal pain and the presence of heart-related issues.")
    sns.displot(df, x="thalach", hue="target", multiple="dodge")
    st.pyplot()
    st.write("The distribution plot (displot) analyzing the maximum heart rate achieved (thalach) concerning the presence or absence of heart disease (target) yields a significant observation. After considering the chest pain type (cp) variable, which is a well-known indicator of heart issues, thalach emerges as a crucial factor closely linked to the target variable.The analysis demonstrates a clear trend: as thalach increases, indicating a higher maximum heart rate achieved during activities, individuals are more susceptible to heart disease.")
    sns.relplot(data=df, x="cp", y="thalach", hue="target")
    st.pyplot()
    st.write("The plot suggests that chest pain type 2 (Non-Anginal pain) is notably associated with heart disease, especially when combined with higher values of thalach (maximum heart rate achieved). Individuals experiencing non-anginal pain (cp=2) and displaying elevated thalach levels are at a substantially higher risk of heart disease. This observation emphasizes the importance of considering both chest pain type and maximum heart rate achieved when evaluating the likelihood of heart-related issues.")
    


    
    








