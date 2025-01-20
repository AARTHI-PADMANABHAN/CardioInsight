import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from sklearn import model_selection
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, auc

st.set_option('deprecation.showPyplotGlobalUse', False)
df = pd.read_csv("heart.xls")
X = np.array(df.drop(['target'], axis=1))
y = np.array(df['target'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, stratify=y, random_state=42, test_size = 0.2)

st.markdown("<h1 style='text-align: center;'>CardioInsight: A Predictive Analytics Tool for Heart Disease</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #999999; font-style: italic;'>Presented by Aarthi Padmanabhan</p>", unsafe_allow_html=True)

st.sidebar.markdown("🌟Hello Explorer! Welcome to the Heart Disease Data Analysis app! Step inside and explore, where insights await at every corner.🎩✨")

# Initialize session state
def init_session_state():
    return {"selected_page": "introduction"}

# Function to show Introduction page
def show_introduction():
    st.title("Introduction")
    tab1, tab2, tab3, tab4 = st.tabs(["Problem Defintion", "Objectives", "Key Features", "Bio",])

    with tab1:
        st.write("In the realm of healthcare and cardiology, the complex nature of heart disease demands a comprehensive understanding of its multifaceted influencing factors. Key variables such as age, gender, chest pain type, resting blood pressure, cholesterol levels, fasting blood sugar, electrocardiographic measurements, maximum heart rate achieved, and exercise-induced angina collectively contribute to the intricate landscape of cardiovascular health. The need to unravel the interplay among these factors is crucial for effective prevention and treatment strategies.")
        
        st.image("heart-disease-hub.jpg", caption="Heart Disease Analysis", use_column_width=True, width=10)
        st.write("Despite advancements in medical science, the pathways to heart disease remain elusive, prompting the initiation of this app to delve into an extensive dataset. Through meticulous analysis, the goal is to unearth hidden correlations and trends, providing a nuanced perspective that can enhance early diagnosis and management of heart-related conditions.")
        st.write("CardioInsight addresses the imperative challenge of navigating the intricate web of variables influencing heart disease. The vast dataset serves as a treasure trove, holding valuable information that, when deciphered, can illuminate the subtle connections between diverse attributes. The overarching problem lies in the complexity of understanding how age, gender, physiological measurements, and lifestyle factors intertwine to contribute to the development of heart-related conditions. The exploration of these intricate relationships aims to unlock insights that have the potential to transform the landscape of cardiovascular care. By visualizing and comprehending these patterns, the project endeavors to bridge the knowledge gap, empowering healthcare professionals with a more nuanced understanding of heart disease etiology for improved prevention and timely intervention.")

    with tab2:
        st.write("The primary objective of CardioInsight is to act as a predictive analytics tool for heart disease, leveraging neural network algorithm to analyze diverse patient attributes. Through this exploration, the project aims to provide valuable insights that can contribute to the early diagnosis and management of heart-related conditions. By visualizing intricate relationships among these factors, CardioInsight strives to offer healthcare professionals a comprehensive and user-friendly platform for making informed decisions in the realm of cardiology.")
        
        st.write("This app aims to explore the Cleveland Heart Disease dataset and answer the below question:")

        # List of questions in italics
        st.markdown("""
           - Risk Factors Identification: What are the primary risk factors associated with heart disease, and how do they interrelate?
        """)
        st.markdown("""
           - Predictive Analysis: Can the tool accurately predict the likelihood of heart disease for an individual based on specific attributes such as age, gender, chest pain type, resting blood pressure, cholesterol levels, fasting blood sugar, electrocardiographic measurements, maximum heart rate achieved, and exercise-induced angina?
        """)

    with tab3:
        st.subheader("Key Features:")
        st.markdown("""
        - **Data Analysis and Visualization:**
          CardioInsight employs advanced data analysis techniques to uncover correlations and trends within the dataset. Visualization tools are utilized to represent complex relationships, making it easier for healthcare professionals to interpret and comprehend the findings.
        
        - **Neural Network Algorithm:**
          CardioInsight app integrates neural network algorithm to predict the likelihood of heart disease based on the input variables. This algorithm continuously learn from the dataset, adapting to new information and enhancing the accuracy of predictions over time.
        
        - **User-Friendly Interface:**
          CardioInsight features an intuitive and user-friendly interface, allowing healthcare professionals to input patient data seamlessly. The platform provides clear and concise results, aiding in quick decision-making and facilitating efficient patient care.
        
        - **Customizable Risk Assessment:**
          CardioInsight allows for customizable risk assessments, enabling healthcare professionals to tailor predictions based on specific patient profiles. This flexibility enhances the utility of CardioInsight across diverse patient populations.
        
        - **Continuous Learning and Improvement:**
          CardioInsight is designed to adapt and improve continuously. As more data becomes available, the app refines its predictive capabilities, staying at the forefront of advancements in cardiology and healthcare.
        """)
        
    with tab4:
        st.write("About me")
        
        st.write("Hello! I am Aarthi, a graduate student in the Data Science program at Michigan State University. Prior to my masters degree, I have two years of experience working as a Software Developer I at Oracle.")
        st.write("An aspiring data analyst with a strong foundation in an array of programming languages, mathematical models and data science methodologies. I enjoy collaborating with corss-functional teams, bringing my technical expertise to the table while valuing diverse perspecties to drive successful project outcomes.")
        
        st.write("Apart from my involvement in data science, I love being out in the nature, listening to music, cooking and enjoy volunteering activites.")
        st.write("Feel free to reach out to me at [aarthi9929@gmail.com] or connect with me on [LinkedIn](https://www.linkedin.com/in/aarthi-padmanabhan-2b47b3183/).")
        

# Function to show Analysis page
def show_data_overview():
    st.title("Data Overview")
    tab1, tab2, tab3 = st.tabs(["About data", "Summary Statistics", "Correlation Matrix",])
    
    with tab1:
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
        
            
    with tab2:
        st.subheader("Statistics of Cleveland heart disease data")
        st.write(df.describe())
        st.write("The descriptive statistics provided above offer valuable insights into the dataset. Upon careful examination, it becomes evident that the dataset is complete and does not contain any missing values. Consequently, the absence of missing values signifies that no preprocessing steps, such as imputation or removal of incomplete records, were necessary. This completeness enhances the dataset's reliability and ensures the accuracy of the analyses and conclusions drawn from it.")
    
    with tab3:
        st.subheader("Correlation matrix of Cleveland heart disease data")
        corr_matrix = df.corr()

        # Create a triangular mask
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Draw the heatmap with Seaborn
        heatmap_fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", vmin=-1, vmax=1, center=0,
                    square=True, linewidths=.5, annot=True, fmt=".2f", ax=ax)
        
        # Display the Seaborn heatmap using st.pyplot
        st.pyplot(heatmap_fig)
        
        # Extract and display annotations separately
        st.subheader("Correlation Matrix Annotations:")
        # Display the Matplotlib figure using st.pyplot
        st.write("It is evident that the attributes—namely, chest pain type (cp), resting electrocardiographic results (restecg), maximum heart rate achieved during exercise (thalach), and the slope of the peak exercise ST segment (slope)—are positively correlated with the occurrence of heart disease when compared to other attributes. Let's examine these attributes in the Exploratory Data Analysis (EDA) section)!")
        
# Function to show Analysis page
def show_eda():
    st.subheader("Risk Factors Identification")
    tab1, tab2, tab3 = st.tabs(["Feature Investigation", "Feature Correlation", "Further Exploration",])
    
    with tab1:
        st.subheader("Value Spread Visualization")
        st.write("This provides a visual representation of the distribution of a dataset. They help you understand how the data "
         "points are spread across different ranges (bins), revealing insights about the central tendency, spread of data, "
         "skewness, etc.")

        st.write("Go ahead and identify the spread of the data with histograms! Use the options on the side column to customize your "
         "analysis.👇")
        
        col1, col2 = st.columns([1,3])
        with col1:
            x_column = st.selectbox("Select X-axis Column:", df.columns, key="x_column")
            bins = st.slider("Select no. of bins:", min_value=1, max_value=50, value=10)
        with col2:
            sns.histplot(data=df, x=x_column, bins=bins, hue="target", multiple="stack", palette="Blues")
            plt.xlabel(x_column)
            plt.ylabel("Frequency")
            plt.title(f'Histogram of {x_column.capitalize()} with {bins} Bins')
            st.pyplot()
        st.subheader("Quantile-Quantile Analysis")
        st.write("Q-Q plots are valuable tools for statisticians and data analysts to quickly assess the distributional properties of a dataset and make informed decisions about the appropriate statistical techniques to apply. Please select a column to customize your analysis.")
        selected_column = st.selectbox("Select a column for Q-Q plot:", df.select_dtypes(include='number').columns)
        # Create a Q-Q plot
        quantiles = sm.ProbPlot(df[selected_column])
        qq_plot = quantiles.qqplot(line='s')
        
        # Customize the plot
        plt.title(f"Q-Q Plot for {selected_column}")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Sample Quantiles")
        
        # Add legends
        plt.legend(["Sample Quantiles", "Theoretical Quantiles"], loc="best")
        
        # Show the plot in Streamlit
        st.pyplot(qq_plot)
        
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
        st.subheader("Bivariate Analysis")
        col1, col2 = st.columns([1,3])
        with col1:
            x = st.selectbox("Select X-axis Column:", df.columns, key="x")
            y = st.selectbox("Select Y-axis Column:", df.columns, key="y")
        with col2:
            df_temp = df.copy()

            # Convert 'target' column to string type in the copy
            df_temp['target'] = df_temp['target'].astype(str)
            
            # Specify a color sequence using color_discrete_sequence
            color_sequence = px.colors.qualitative.Set1
            
            # Specify colors for each class using color_discrete_map
            color_map = {'0': 'blue', '1': 'red'}
            
            fig = px.scatter(df_temp, x=x, y=y, color='target',
                             labels={x: x.capitalize(),
                                     y: y.capitalize(),
                                     'target': 'Heart Disease'},
                             title=f"2D Scatter Plot: {x.capitalize()} vs {y.capitalize()}",
                             color_discrete_sequence=color_sequence,
                             color_discrete_map=color_map,
                             hover_data=['age']) 
            # Display the scatter plot
            st.plotly_chart(fig)
        st.write("Hover over the data points to view the age information.")
        st.subheader("Cross-Variable Scatter Matrix")
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
        st.subheader("Multivariate Analysis")
        st.write("For deeper insights and a more comprehensive understanding of the data relationships, consider using 3D plots. They can reveal intricate patterns and correlations between multiple variables simultaneously.")
        X = st.selectbox("Select X-axis Column:", df.columns, key="X")
        Y = st.selectbox("Select Y-axis Column:", df.columns,  key="Y")
        Z = st.selectbox("Select Z-axis Column:", df.columns, key="Z")
    
        # Specify a color sequence using color_discrete_sequence
        color_sequence = px.colors.qualitative.Set1
        
        # Specify colors for each class using color_discrete_map
        color_map = {'0': 'blue', '1': 'red'}
        fig = px.scatter_3d(df_temp, x=X, y=Y, z=Z, color='target',
                        labels={X: X,
                                Y: Y,
                                Z: Z,},
                        title="3D Scatter Plot with Hue as Target",
                        color_discrete_sequence=color_sequence,
                        color_discrete_map=color_map,
                        hover_data=['age'])  # You can add more hover_data as needed
    
        # Display the 3D scatter plot
        st.plotly_chart(fig)
        st.write("Zoom in/out, hover over the data for better visualization.")

def create_binary_model(num_neurons_layer1, num_neurons_layer2, dropout_rate,
                        learning_rate, activation_function, optimizer, regularization_strength,
                        include_dropout=True):
    # Create model
    model = Sequential()
    model.add(Dense(num_neurons_layer1, input_dim=13, kernel_initializer='normal',
                    kernel_regularizer=regularizers.l2(regularization_strength),
                    activation=activation_function))
    
    if include_dropout:
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(num_neurons_layer2, kernel_initializer='normal',
                    kernel_regularizer=regularizers.l2(regularization_strength),
                    activation=activation_function))
    
    if include_dropout:
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    optimizer_instance = Adam(lr=learning_rate) if optimizer == 'adam' else optimizer
    model.compile(loss='binary_crossentropy', optimizer=optimizer_instance, metrics=['accuracy'])
    
    return model

def show_model_accuracy(num_neurons_layer1, num_neurons_layer2, dropout_rate,
                        learning_rate, activation_function, optimizer, regularization_strength, epochs,
                        batch_size, include_dropout=True):
    binary_model = create_binary_model(num_neurons_layer1, num_neurons_layer2, dropout_rate,
                           learning_rate, activation_function, optimizer, regularization_strength,
                           include_dropout)
    history = binary_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
    
    y_prob = binary_model.predict(X_test)
    y_pred = (y_prob > 0.5).astype(int)  
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Display confusion matrix using Plotly heatmap
    st.subheader("Confusion Matrix")
    
    colorscale = [
            [0, 'rgb(255,255,255)'],
            [0.5, 'rgb(173,216,230)'],  # Light Blue
            [1, 'rgb(0,128,128)']      # Teal
        ]
    
    fig = ff.create_annotated_heatmap(z=cm, x=["Predicted 0", "Predicted 1"],
                                          y=["Actual 0", "Actual 1"],
                                          colorscale=colorscale)
    fig.update_layout(width=550, height=300)
    st.plotly_chart(fig)
    
    df_accuracy = pd.DataFrame({
        'Epoch': np.arange(1, epochs + 1),
        'Train Accuracy': history.history['accuracy'],
        'Test Accuracy': history.history['val_accuracy']
    })
    
    fig_accuracy = go.Figure()
    
    fig_accuracy.add_trace(go.Scatter(x=df_accuracy['Epoch'], y=df_accuracy['Train Accuracy'],
                                      mode='lines', name='Train Accuracy'))
    fig_accuracy.add_trace(go.Scatter(x=df_accuracy['Epoch'], y=df_accuracy['Test Accuracy'],
                                      mode='lines', name='Test Accuracy'))
    
    # Set the size of the graph
    fig_accuracy.update_layout(title='Model Accuracy Over Epochs',
                               xaxis_title='Epoch',
                               yaxis_title='Accuracy',
                               legend=dict(x=0, y=1, traceorder='normal'),
                               width=550,  # Set the width of the graph
                               height=400)  # Set the height of the graph
    
    st.plotly_chart(fig_accuracy)
    
    # Interactive plot using Plotly for Model Loss
    df_loss = pd.DataFrame({
        'Epoch': np.arange(1, epochs + 1),
        'Train Loss': history.history['loss'],
        'Test Loss': history.history['val_loss']
    })

    fig_loss = go.Figure()
    
    fig_loss.add_trace(go.Scatter(x=df_loss['Epoch'], y=df_loss['Train Loss'],
                                  mode='lines', name='Train Loss'))
    fig_loss.add_trace(go.Scatter(x=df_loss['Epoch'], y=df_loss['Test Loss'],
                                  mode='lines', name='Test Loss'))
    
    # Set the size of the graph
    fig_loss.update_layout(title='Model Loss Over Epochs',
                           xaxis_title='Epoch',
                           yaxis_title='Loss',
                           legend=dict(x=0, y=1, traceorder='normal'),
                           width=550,  # Set the width of the graph
                           height=400)  # Set the height of the graph
    
    st.plotly_chart(fig_loss)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Display precision and recall
    st.subheader("Precision and Recall")
    st.write(f"- **Precision:** {precision:.4f}")
    st.write(f"- **Recall:** {recall:.4f}")

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Display AUC-ROC curve using Plotly
    st.subheader("AUC-ROC Curve")
    fig_auc_roc = go.Figure()
    fig_auc_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve'))
    fig_auc_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
    fig_auc_roc.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate'),
        width=600,
        height=400
    )
    st.plotly_chart(fig_auc_roc)

def preprocess(sex, cp, exang,fbs,restecg):   
 
    if sex=="male":
        sex=1 
    else: 
        sex=0
    
    if cp=="Typical angina":
        cp=0
    elif cp=="Atypical angina":
        cp=1
    elif cp=="Non-anginal pain":
        cp=2
    elif cp=="Asymptomatic":
        cp=2
    
    if exang=="Yes":
        exang=1
    elif exang=="No":
        exang=0
 
    if fbs=="Yes":
        fbs=1
    elif fbs=="No":
        fbs=0
        
    if restecg=="Normal":
        restecg=0
    elif restecg=="ST-T Wave abnormality":
        restecg=1
    elif restecg=="Possible or definite left ventricular hypertrophy":
        restecg=2

    return sex, cp, exang,fbs,restecg
    
def predict_heart_disease():

    st.write(
    "Let's unveil the mystery of your heart health! 🌟 Enter the details below, and the app tells you if your heart is singing a happy tune or if it needs a little extra care:"
    )
    age = st.number_input('Age of persons (29 - 77): ', min_value=29, max_value=77, value=29, step=1)
    sex = st.radio("Select Gender: ", ('male', 'female'))
    cp = st.selectbox('Chest Pain Type', ("Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"))
    trtbps = st.number_input('Resting blood pressure (94 - 200): ', min_value=94, max_value=200, value=94, step=1)
    chol = st.number_input('Serum cholestrol in mg/dl (126 - 564): ', min_value=126, max_value=564, value=126, step=1)
    fbs = st.radio("Fasting Blood Sugar higher than 120 mg/dl", ['Yes', 'No'])
    restecg = st.selectbox('Resting Electrocardiographic Results', ("Normal", "ST-T Wave abnormality", "Possible or definite left ventricular hypertrophy"))
    thalachh = st.number_input('Maximum heart rate achieved thalach (71 - 202): ', min_value=71, max_value=202, value=71, step=1)
    exang = st.selectbox('Exercise Induced Angina', ["Yes", "No"])
    oldpeak = st.number_input(' ST depression induced by exercise relative to rest (oldpeak) (0 - 6.2): ')
    slope = st.number_input(' Slope of the Peak Exercise ST Segment (slope) (0-2): ')
    ca = st.number_input(' Number of major vessels colored by fluoroscopy (0-3): ')
    thal = st.number_input(' Number of major vessels colored by fluoroscopy (1-3): ')
    
    sex, cp, exang, fbs, restecg = preprocess(sex, cp, exang,fbs,restecg)
    
    data= {'age':age, 'sex':sex, 'cp':cp, 'trestbps':trtbps, 'chol':chol, 'fbs':fbs, 'restecg':restecg, 'thalach':thalachh,
       'exang':exang, 'oldpeak':oldpeak, 'slope':slope, 'ca':ca, 'thal':thal
        }
    features = pd.DataFrame(data, index=[0])
    
    if st.button("Predict"):
        # Perform model prediction using the stored user inputs

        binary_model = create_binary_model(16, 8, 0.25, 0.001, 'relu', 'adam', 0.001, False)
        # history = binary_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=175, batch_size=10)
        # Make predictions
        prediction = binary_model.predict(features)
        result = prediction[0]

        # Display prediction
        if result:
            st.success("The model predicts that you have heart disease.")
        else:
            st.success("The model predicts that you do not have heart disease.")
        
def show_model_exploration_classification():
    tab1, tab2 = st.tabs(["Model Analysis: Neural Network", "Heart Disease Classification",])
    with tab1:
        st.subheader("Neural Network Model")
        st.write("👇 Explore with the options below and dive into a detailed analysis of the neural model.")
        st.write("Upon investigation, the following key insights are available in the app:")

        # Add specific details or links to access confusion matrix, model accuracy, and test loss visualizations.
        st.write("- **Confusion Matrix:** View detailed predictions for each class.")
        st.write("- **Model Accuracy:** Track learning progress with accuracy trends.")
        st.write("- **Test Loss:** Evaluate model generalization and potential overfitting.")
        st.write("- **Precision:** Assess the accuracy of positive predictions.")
        st.write("- **Recall:** Evaluate the model's ability to capture all positive instances.")
        st.write("- **AUC-ROC Curve:** Visualize the trade-off between true positive rate and false positive rate.")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            activation_function = st.selectbox("Activation Function", ["relu", "sigmoid"])
        
        with col2:
            num_neurons_layer1 = st.slider("Neurons in Layer 1", min_value=1, max_value=128, value=16)
        
        with col3:
            num_neurons_layer2 = st.slider("Neurons in Layer 2", min_value=1, max_value=128, value=8)
        
        with col4:
            dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
        
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            learning_rate = st.slider("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001)
        
        with col6:
            optimizer = st.selectbox("Optimizer", ["adam", "rmsprop", "sgd"])
        
        with col7:
            regularization_strength = st.slider("L2 Regularization", min_value=0.0, max_value=0.1, value=0.001, step=0.0001)
        
        with col8:
            epochs = st.slider("Epochs", min_value=1, max_value=200, value=180)
        
        batch_size = st.slider("Batch Size", min_value=1, max_value=256, value=10)
            
        include_dropout = st.checkbox("Include Dropout", value=True)
        
        if st.button("Train and Analyze Model"):
            show_model_accuracy(num_neurons_layer1, num_neurons_layer2, dropout_rate,
                                        learning_rate, activation_function, optimizer, regularization_strength, epochs,
                                        batch_size, include_dropout)
            
    with tab2:
        predict_heart_disease()
        
    

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
        st.write("## Next Steps")

        st.write("1. **Continuous Monitoring and Maintenance:** Establish a system for continuous monitoring of model performance in the production environment. Set up alerts "
             "for potential issues and plan for regular model updates or retraining to adapt to changing data patterns.")
        
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
    "[🔧 Model Exploration and Classification](#model-exploration-classification)",
    "[🎓 Conclusion](#conclusion)"
    ])

    # Change selected page based on user click
    if "Introduction" in selected_option:
        session_state["selected_page"] = "introduction"
    elif "Data Overview" in selected_option:
        session_state["selected_page"] = "dataoverview"   
    elif "Exploratory Data Analysis (EDA)" in selected_option:
        session_state["selected_page"] = "eda"
    elif "Model Exploration and Classification" in selected_option:
        session_state["selected_page"] = "model-exploration-classification"    
    elif "Conclusion" in selected_option:
        session_state["selected_page"] = "conclusion"

    # Display the selected page content
    if session_state["selected_page"] == "introduction":
        show_introduction()
    elif session_state["selected_page"] == "dataoverview":
        show_data_overview()    
    elif session_state["selected_page"] == "eda":
        show_eda()
    elif session_state["selected_page"] == "model-exploration-classification":
        show_model_exploration_classification()
    elif session_state["selected_page"] == "conclusion":
        show_conclusion()

# Run the app
if __name__ == "__main__":
    main()

    
     
   
        

    
