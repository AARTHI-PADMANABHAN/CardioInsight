import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm

st.set_option('deprecation.showPyplotGlobalUse', False)
df = pd.read_csv("heart.xls")

st.markdown("<h1 style='text-align: center;'>Heart Disease Data Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #999999; font-style: italic;'>CMSE Mid-Term Project, presented by Aarthi Padmanabhan</p>", unsafe_allow_html=True)

st.sidebar.markdown("🌟Hello Explorer! Welcome to the Heart Disease Data Analysis app! Step inside and explore, where insights await at every corner.🎩✨")

# Initialize session state
def init_session_state():
    return {"selected_page": "introduction"}

# Function to show Introduction page
def show_introduction():
    st.title("Introduction")
    tab1, tab2, tab3 = st.tabs(["Problem Defintion", "Objectives", "About Data",])

    with tab1:
        st.image("Heart_image.png", caption="Heart Disease Analysis", use_column_width=True, width=10)
        description_lines = [
    "In the realm of healthcare and cardiology, understanding the factors influencing heart disease is crucial for prevention and effective treatment.",
    "Various aspects, such as age, gender, chest pain type, resting blood pressure, cholesterol levels, fasting blood sugar, electrocardiographic measurements, maximum heart rate achieved, exercise-induced angina, and several other attributes, play pivotal roles.",
    "Analyzing the extensive dataset on heart disease allows us to explore intricate patterns and relationships among these factors.",
    "This project aims to delve into the depths of this data, seeking correlations and trends that might illuminate the pathways to heart disease.",
    "By visualizing these intricate relationships, we hope to uncover valuable insights that can aid in the early diagnosis and management of heart-related conditions."
     ]
        description = "\n".join(description_lines)
        st.write(description)

    with tab2:
        st.write("This app aims to explore the Cleveland Heart Disease dataset and answer the below question:")

        # List of questions in italics
        st.markdown("""
           - Risk Factors Identification: What are the primary risk factors associated with heart disease, and how do they interrelate?
        """)

    with tab3:
        st.subheader("Overview of Cleveland Heart Disease Data")
        st.write("The Cleveland Heart Disease dataset provides a comprehensive insight into various aspects of patients' health, aiding researchers and healthcare professionals in understanding heart-related conditions. Here's an overview of the key attributes within the dataset:")
        st.markdown("""
            - **Age:** Represents the age of the patient in years.
            - **Sex:** Indicates the gender of the patient (1 = male, 0 = female).
            - **Chest Pain Type (cp):** Describes the type of chest pain experienced by the patient.
              - 3: Typical Angina - Chest pain related to decreased blood supply to the heart, relieved by rest or nitroglycerin.
              - 2: Non-Anginal Pain - Chest pain not related to angina.
              - 1: Atypical Angina - Chest pain not typical angina, related to decreased blood supply to the heart.
              - 0: Asymptomatic - Individuals who do not experience chest pain.
            - **Resting Blood Pressure (trestbps):** Resting blood pressure in mm Hg on admission to the hospital.
            - **Serum Cholestoral (chol):** Serum cholesterol in mg/dl.
            - **Fasting Blood Sugar (fbs):** Fasting blood sugar level > 120 mg/dl (1 = true, 0 = false).
            - **Resting Electrocardiographic Results (restecg):** Resting electrocardiographic results.
              - 0: Normal - Normal results.
              - 1: ST-T Wave Abnormality - Abnormality related to ST-T wave.
              - 2: Left Ventricular Hypertrophy - Showing probable or definite left ventricular hypertrophy.
            - **Maximum Heart Rate Achieved (thalach):** The person's maximum heart rate achieved.
            - **Exercise Induced Angina (exang):** Exercise-induced angina (1 = yes, 0 = no).
            - **ST Depression Induced by Exercise Relative to Rest (oldpeak):** ST depression induced by exercise relative to rest.
            - **Slope of the Peak Exercise ST Segment (slope):** The slope of the peak exercise ST segment.
              - 0: Upsloping - Sloping upwards indicates a better prognosis.
              - 1: Flat - Flat indicates no upsloping or downsloping of the ST segment.
              - 2: Downsloping - Sloping downwards indicates an unhealthy heart.
            - **Number of Major Vessels (ca):** Number of major vessels colored by fluoroscopy (0-3).
            - **Thalassemia (thal):** A blood disorder called thalassemia.
              - 1: Normal - Normal thalassemia.
              - 2: Fixed Defect - Fixed defect thalassemia.
              - 3: Reversible Defect - Reversible defect thalassemia.
        """)
        
        st.subheader("**Dataset Source:**")
        st.write("The Cleveland Heart Disease dataset was downloaded from the Kaggle website.")
        st.write("You can find the dataset and more information at the following link: \n [Cleveland Heart Disease Dataset](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland)")

# Function to show Analysis page
def show_data_overview():
    st.title("Data Overview")
    tab1, tab2, tab3 = st.tabs(["Display data", "Summary Statistics", "Correlation Matrix",])
    
    with tab1:
        st.subheader("Cleveland Heart Disease Data")
        filter_option = st.selectbox("Select Filtering Option", ["Filter by Age Range", "Display First 10 Rows", "Display Last 10 Rows", "Display Full data"])

        # Filter data based on user selection
        if filter_option == "Filter by Age Range":
            age_range_start, age_range_end = st.slider("Select Age Range", df['age'].min(), df['age'].max(), (df['age'].min(), df['age'].max()))
            filtered_df = df[(df['age'] >= age_range_start) & (df['age'] <= age_range_end)]
            st.subheader(f"Filtered Data for Age Range: {age_range_start} to {age_range_end}")
            st.write(filtered_df)
        elif filter_option == "Display First 10 Rows":
            st.subheader("First 10 Rows of Data")
            st.write(df.head(10))
        elif filter_option == "Display Last 10 Rows":
            st.subheader("Last 10 Rows of Data")
            st.write(df.tail(10))
        elif filter_option == "Display Full data":
            st.subheader("Full Data")
            st.write(df)
            
    with tab2:
        st.subheader("Statistics of Cleveland heart disease data")
        st.write(df.describe())
        st.write("The descriptive statistics provided above offer valuable insights into the dataset. Upon careful examination, it becomes evident that the dataset is complete and does not contain any missing values. Consequently, the absence of missing values signifies that no preprocessing steps, such as imputation or removal of incomplete records, were necessary. This completeness enhances the dataset's reliability and ensures the accuracy of the analyses and conclusions drawn from it.")
    
    with tab3:
        st.subheader("Correlation matrix of Cleveland heart disease data")
        st.write(df.corr())
        st.write("It is evident that the attributes—namely, chest pain type (cp), resting electrocardiographic results (restecg), maximum heart rate achieved during exercise (thalach), and the slope of the peak exercise ST segment (slope)—are positively correlated with the occurrence of heart disease when compared to other attributes. Let's examine these attributes in the Exploratory Data Analysis (EDA) section)!")
        
# Function to show Analysis page
def show_eda():
    st.subheader("Risk Factors Identification")
    tab1, tab2, tab3 = st.tabs(["Feature Investigation", "Feature Correlation", "Further Exploration",])
    
    with tab1:
        st.subheader("Histogram plot")
        st.write("Histograms provide a visual representation of the distribution of a dataset. They help you understand how the data "
         "points are spread across different ranges (bins), revealing insights about the central tendency, spread of data, "
         "skewness, etc.")

        st.write("Go ahead and identify the spread of the data with histograms! Use the options on the side column to customize your "
         "analysis.👇")
        
        col1, col2 = st.columns([1,3])
        with col1:
            x_column = st.selectbox("Select X-axis Column:", df.columns, key="x_column")
            bins = st.slider("Select no. of bins:", min_value=1, max_value=50, value=10)
        with col2:
            sns.histplot(data=df, x=x_column, hue='target', bins=bins, color='skyblue', edgecolor='black')
            plt.xlabel(x_column)
            plt.ylabel("Frequency")
            plt.title(f'Histogram of {x_column.capitalize()} with {bins} Bins')
            st.pyplot()
        st.subheader("Q-Q Plot")
        st.write("Q-Q plots are valuable tools for statisticians and data analysts to quickly assess the distributional properties of a dataset and make informed decisions about the appropriate statistical techniques to apply. Please select a column to customize your analysis.")
        selected_column = st.selectbox("Select a column for Q-Q plot:", df.select_dtypes(include='number').columns)
        quantiles = sm.ProbPlot(df[selected_column]).qqplot(line='s')
        plt.title(f"Q-Q Plot for {selected_column}")
        plt.title(f"Q-Q Plot for {selected_column}")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Sample Quantiles")
        st.pyplot(quantiles)
        
        st.subheader("Major Insights:")
        st.markdown("""
        - The incidence of heart disease is more prevalent among individuals aged between 53 to 68.
        - Similarly, the occurrence of heart disease is higher in males compared to females.
        - Individuals experiencing chest pain type 2 are more prone to heart disease.
        - High cholesterol levels do not necessarily guarantee heart disease; it depends on other contributing factors.
        - A higher slope of the peak exercise ST segment is indicative of heart disease.
        """)
         
    with tab2:
        st.write("Let's visualize relationship between multiple variables and figure out patterns, clusters and correlations using PairPlot.")
        st.subheader("2D Scatter Plot")
        col1, col2 = st.columns([1,3])
        with col1:
            x = st.selectbox("Select X-axis Column:", df.columns, key="x")
            y = st.selectbox("Select Y-axis Column:", df.columns, key="y")
        with col2:
            fig = px.scatter(df, x=x, y=y, color='target',
                     labels={x: x.capitalize(),
                             y: y.capitalize(),
                             'target': 'Heart Disease'},
                     title=f"2D Scatter Plot: {x.capitalize()} vs {y.capitalize()}",
                     color_continuous_scale=px.colors.qualitative.Set1,
                     hover_data=['age'])  # You can add more hover_data as needed
            # Display the scatter plot
            st.plotly_chart(fig)
        st.write("Hover over the data points to view the age information.")
        st.subheader("PairPlot")
        selected_features = st.multiselect("Select Features:", df.columns, default=['cp', 'thalach', 'target'])
        if selected_features:
            pair_plot = sns.pairplot(data=df, vars=selected_features, hue='target', palette='Set1', diag_kind='kde')
            pair_plot.fig.suptitle("Pair Plot for Selected Features with Target Variable", y=1.02)
            plt.show()
            st.pyplot()
        else:
            st.warning("Please select at least one feature to generate the pair plot.")
        st.subheader("Major insights:")
        st.markdown("- **Chest Pain Type (cp):** Positive correlation with heart disease, especially type '2'.")
        st.markdown("- **Resting Electrocardiographic Results (restecg):** Positive correlation with heart disease.")
        st.markdown("- **Maximum Heart Rate Achieved (thalach):** Positive correlation with heart disease, higher values indicate higher risk.")
        st.markdown("- **Slope of the Peak Exercise ST Segment (slope):** Positive correlation with heart disease.")
        st.markdown("- **High thalach with cp '2':** Individuals with chest pain type '2' and high thalach are more likely to have heart disease.")
            
    with tab3:
        st.subheader("3D Plot")
        st.write("For deeper insights and a more comprehensive understanding of the data relationships, consider using 3D plots. They can reveal intricate patterns and correlations between multiple variables simultaneously.")
        X = st.selectbox("Select X-axis Column:", df.columns, key="X")
        Y = st.selectbox("Select Y-axis Column:", df.columns,  key="Y")
        Z = st.selectbox("Select Z-axis Column:", df.columns, key="Z")
    
        fig = px.scatter_3d(df, x=X, y=Y, z=Z, color='target',
                        labels={X: X,
                                Y: Y,
                                Z: Z,},
                        title="3D Scatter Plot with Hue as Target",
                        color_continuous_scale=px.colors.qualitative.Set1,
                        hover_data=['age'])  # You can add more hover_data as needed
    
        # Display the 3D scatter plot
        st.plotly_chart(fig)
        st.write("Zoom in/out, hover over the data for better visualization.")

# Function to show Conclusion page
def show_conclusion():
    tab1, tab2, tab3 = st.tabs(["Insights", "Next Steps", "References",])
    with tab1:
        st.subheader("Conclusions that can be drawn from observations are:")
        st.markdown("### 1. Elevated Probability of Heart Disease in Specific Age Group")
        st.markdown("There exists a specific age group within our dataset where the probability of experiencing heart disease is markedly elevated. Upon a closer analysis of the KDE plot depicting the relationship between age and the likelihood of heart disease, it appears that there is no discernible threshold or significant correlation between age and the target variable. Other factors such as chest pain type, the person's maximum heart rate achieved, and genetic predisposition could potentially play more significant roles in influencing heart health outcomes.")
        
        st.markdown("### 2. Highly correlated variables")
        st.markdown("It is crucial to note that, among the various attributes, chest pain type (cp) and maximum heart rate achieved (thalach) exhibit a notably high correlation with heart disease. These variables serve as strong indicators of an individual's susceptibility to heart-related conditions, making them crucial focal points for further analysis.")
        
        st.markdown("### 3. Correlation Between Chest Pain Type and Heart Disease")
        st.markdown("Specifically, individuals with chest pain type 2 (Non-anginal pain) appear to be more susceptible to heart disease, as they are the predominant group affected within this dataset. The higher count of heart disease cases among individuals experiencing this specific chest pain type suggests a potential correlation between non-anginal pain and the presence of heart-related issues.")
        
        st.markdown("### 4. Influence of Maximum Heart Rate Achieved on Heart Disease")
        st.markdown("After considering the chest pain type (cp) variable, which is a well-known indicator of heart issues, thalach emerges as a crucial factor closely linked to the target variable. The analysis demonstrates a clear trend: as thalach increases, indicating a higher maximum heart rate achieved during activities, individuals are more susceptible to heart disease.")
        
        st.markdown("### 5. Combined Influence of Chest Pain Type and Maximum Heart Rate Achieved")
        st.markdown("Individuals experiencing non-anginal pain (cp=2) and displaying elevated thalach levels are at a substantially higher risk of heart disease. This observation emphasizes the importance of considering both chest pain type and maximum heart rate achieved when evaluating the likelihood of heart-related issues.")
    
    with tab2:
        st.write("## Next Steps After EDA:")

        st.write("After conducting a thorough exploratory data analysis (EDA) on the Cleveland Heart Disease dataset, "
                 "you've gained valuable insights into the relationships between various attributes and the presence of heart disease. "
                 "Now, it's time to move forward and leverage this understanding to build predictive models and gain deeper insights. "
                 "Here are the next steps you can consider:")
        
        st.write("1. **Feature Engineering:** Based on the insights gained from EDA, you can engineer new features or transform "
                 "existing ones to enhance the predictive power of your models. For example, you can create interaction features between "
                 "attributes, derive new categorical variables, or normalize numerical features for consistent scaling.")
        
        st.write("2. **Machine Learning Modeling:** Implement machine learning algorithms to predict the likelihood of heart disease. "
                 "Common models like Logistic Regression, Decision Trees, Random Forest Classifier, Support Vector Machines, k-Nearest Neighbors, "
                 "Naive Bayes, and Neural Networks can be employed. Experiment with different models, tune hyperparameters, and assess their "
                 "performance using metrics like accuracy, precision, recall, and F1-score.")
        
    with tab3:
        st.write("## References:")
        
        st.write("- [Cleveland Heart Disease Dataset](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland)")
        st.write("- [Stream Lit App Documentation](https://docs.streamlit.io/library/get-started)")
        
# Main function to handle the app layout
def main():
    # Initialize session state
    session_state = init_session_state()

    st.sidebar.title("Menu")
    
    # Create clickable hyperlinks for each page
    selected_option = st.sidebar.radio("", [
    "[📚 Introduction](#introduction)",
    "[🔍 Data Overview](#dataoverview)",
    "[📊 Exploratory Data Analysis (EDA))](#eda)",
    "[🎓 Conclusion](#conclusion)"
    ])

    # Change selected page based on user click
    if "Introduction" in selected_option:
        session_state["selected_page"] = "introduction"
    elif "Data Overview" in selected_option:
        session_state["selected_page"] = "dataoverview"   
    elif "Exploratory Data Analysis (EDA)" in selected_option:
        session_state["selected_page"] = "eda"
    elif "Conclusion" in selected_option:
        session_state["selected_page"] = "conclusion"

    # Display the selected page content
    if session_state["selected_page"] == "introduction":
        show_introduction()
    elif session_state["selected_page"] == "dataoverview":
        show_data_overview()    
    elif session_state["selected_page"] == "eda":
        show_eda()
    elif session_state["selected_page"] == "conclusion":
        show_conclusion()

# Run the app
if __name__ == "__main__":
    main()
