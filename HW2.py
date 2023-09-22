import seaborn as sns
import streamlit as st
import plotly.express as px
df_iris = sns.load_dataset("iris")
three_dim = px.scatter_3d(
    df_iris,
    x='sepal_length',
    y='sepal_width',
    z='petal_width',
    color='species',
    title='3D Scatter Plot of Iris Dataset'
)
three_dim.update_layout(scene=dict(xaxis_title='Sepal Length', yaxis_title='Sepal Width', zaxis_title='Petal Width'))
st.plotly_chart(three_dim)
st.write("""
The iris dataset consists of four measurements (length and width of petals and sepals). This dataset contains species of type: setosa, virginica, and versicolor. The 3D scatter plot of iris dataset is created to visualize the connection between sepal length, sepal width and petal width.
""")
