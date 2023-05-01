import streamlit as st
# import webbrowser

# url = 'https://archive.ics.uci.edu/ml/index.php'

# if st.button('Open browser'):
    # webbrowser.open_new_tab(url)
   

import streamlit as st
import plotly_express as px
import pandas as pd
import subprocess
#for encoding
from sklearn.preprocessing import LabelEncoder
#for train test split
from sklearn.model_selection import train_test_split

# Setup file upload
uploaded_file = st.sidebar.file_uploader(
                        label="Upload your CSV or Excel file.",
                         type=['csv', 'xlsx'])

global df

if uploaded_file is not None:
    print(uploaded_file)
    print("hello")

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        print(e)
        df = pd.read_excel(uploaded_file)
        
global numeric_columns
global non_numeric_columns
try:
    st.write(df)
    numeric_columns = list(df.select_dtypes(['int64','float64','int32','float32','int','float']).columns)
    non_numeric_columns = list(df.select_dtypes(['object','bool']).columns)
    non_numeric_columns.append(None)
    print(non_numeric_columns)
except Exception as e:
    print(e)
    st.write("Please upload file to the application.")

st.header("Random Forest Classification")


options = st.multiselect(
     'What are your favorite colors',
     ['Green', 'Yellow', 'Red', 'Blue'],
     ['Yellow', 'Red'])


# col1, col2, col3 = st.columns(3)

# with col1:
    # st.header("A cat")
    # st.image("https://static.streamlit.io/examples/cat.jpg")

# with col2:
    # st.header("A dog")
    # st.image("https://static.streamlit.io/examples/dog.jpg")

# with col3:
    # st.header("An owl")
    # st.image("https://static.streamlit.io/examples/owl.jpg")

col = df.columns    

st.write(col)



mul = st.multiselect(
     'What is your X variable name',
     options=df.columns)
sel = st.selectbox('What is your Target Variable',options=non_numeric_columns)

#df['mul']


X = []
X.append(mul)
 
y = []
y.append(sel)

df1 = pd.DataFrame({"X variable":X,"target":y})

st.write(df1)


#ir = df['X']
     
#st.write(df['X'])
     
    
# data split# Split dataset into training set and test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

