import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as pre
from sklearn import linear_model 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import sklearn.linear_model as glm
from sklearn.metrics import mean_squared_error, r2_score

st.set_option('deprecation.showfileUploaderEncoding', False)

# title of the app
st.title("Data Classification")

import base64

main_bg = "bg.jpg"
main_bg_ext = "jpg"
st.write(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# Add a sidebar
st.sidebar.subheader("Upload the Dataset")

# Setup file upload
uploaded_file = st.sidebar.file_uploader(
                        label="Upload your CSV or Excel file",
                         type=['csv', 'xlsx'])
                         
default_dataset = st.sidebar.selectbox(
    label = "select the dataset",
    options=['None', 'Campus Requirement Dataset', 'Carseat Dataset', 'Cereal Dataset', 'Insurance Dataset', 'Iris Dataset', 'Mtcars Dataset', 'Penguin Dataset', 'Pokemon Dataset','Students Dataset','Students Test Performance Dataset']
)

global df

st.subheader((default_dataset))

if default_dataset == 'Campus Requirement Dataset':
    try:
        df = pd.read_csv("D:/Project Dataset/Campus Requirement.csv")
    except Exception as e:
        print(e)   

if default_dataset == 'Carseat Dataset':
    try:
        df = pd.read_csv("D:/Project Dataset/carseats.csv")
    except Exception as e:
        print(e)  

if default_dataset == 'Cereal Dataset':
    try:
        df = pd.read_csv("D:/Project Dataset/cereal.csv")
    except Exception as e:
        print(e)  

if default_dataset == 'Insurance Dataset':
    try:
        df = pd.read_csv("D:/Project Dataset/insurance.csv")
    except Exception as e:
        print(e)  

if default_dataset == 'Iris Dataset':
    try:
        df = pd.read_csv("D:/Project Dataset/IRIS.csv")
    except Exception as e:
        print(e)         

if default_dataset == 'Mtcars Dataset':
    try:
        df = pd.read_csv("D:/Project Dataset/mtcars.csv")
    except Exception as e:
        print(e)

if default_dataset == 'Penguin Dataset':
    try:
        df = pd.read_csv("D:/Project Dataset/Penguins_data.csv")
    except Exception as e:
        print(e)

if default_dataset == 'Pokemon Dataset':
    try:
        df = pd.read_csv("D:/Project Dataset/Pokemon.csv")
    except Exception as e:
        print(e)

if default_dataset == 'Students Dataset':
    try:
        df = pd.read_csv("D:/Project Dataset/students.csv")
    except Exception as e:
        print(e)
        
if default_dataset == 'Students Test Performance Dataset':
    try:
        df = pd.read_csv("D:/Project Dataset/StudentsPerformance.csv")
    except Exception as e:
        print(e)
                     


if uploaded_file is not None:
    print(uploaded_file)
    print("hello")

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        print(e)
        df = pd.read_excel(uploaded_file)
 
#st.subheader("1.Data")
global numeric_columns
global non_numeric_columns
try:
    st.write(df)
    numeric_columns = list(df.select_dtypes(['float','int','float32','int32','float64','int64']).columns)
    non_numeric_columns = list(df.select_dtypes(['object','bool']).columns)
    non_numeric_columns.append(None)
    print(non_numeric_columns)
except Exception as e:
    print(e)
    st.write("Please upload file to the application.")


df_new = df.dropna()
le = pre.LabelEncoder()
for x in df_new.select_dtypes(include = 'object').columns.tolist():
    print(x)
    df_new[x]= le.fit_transform(df_new[x])
    
X = df_new.iloc[:,:-1] # Using all column except for the last column as X
Y = df_new.iloc[:,-1] # Selecting the last column as Y        # Selecting the last column as Y
# Data splitting
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100,random_state = 12)

st.markdown('**1.2. Data splits**')
st.write('Training set')
st.info(X_train.shape)
st.write('Test set')
st.info(X_test.shape)

st.markdown('**1.3. Variable details**:')
st.write('X variable')
st.info(list(X.columns))
st.write('Y variable')
st.info(Y.name)
    
st.sidebar.subheader('3.Learning Parameters')
parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)
st.sidebar.subheader('2.2. General Parameters')
parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])


rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
random_state=parameter_random_state,
max_features=parameter_max_features,
criterion=parameter_criterion,
min_samples_split=parameter_min_samples_split,
min_samples_leaf=parameter_min_samples_leaf,
bootstrap=parameter_bootstrap,
oob_score=parameter_oob_score,
n_jobs=parameter_n_jobs)
rf.fit(X_train, Y_train)

st.subheader('2. Model Performance')

st.markdown('**2.1. Training set**')
Y_pred_train = rf.predict(X_train)
st.write('Coefficient of determination ($R^2$):')
st.info( r2_score(Y_train, Y_pred_train) )

st.write('Error (MSE or MAE):')
st.info( mean_squared_error(Y_train, Y_pred_train) )

st.markdown('**2.2. Test set**')
Y_pred_test = rf.predict(X_test)
st.write('Coefficient of determination ($R^2$):')
st.info( r2_score(Y_test, Y_pred_test) )

st.write('Error (MSE or MAE):')
st.info( mean_squared_error(Y_test, Y_pred_test) )

st.subheader('3. Model Parameters')
st.write(rf.get_params())

st.subheader('4.Model Score')
st.write("Return the mean accuracy on the given test data and labels.")
st.write(rf.score(X_test,Y_test))

