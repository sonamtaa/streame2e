#simple streamlit application to check the accuracy, confusion_matrix, roc curve and
# classification report on different model with the same datasets

import seaborn as sns
import pandas as pd
import streamlit as st
import joblib
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.metrics import roc_auc_score,roc_curve,auc
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder

# Set Page Layout
st.set_page_config(layout='wide')

df = sns.load_dataset('tips')

options = ['None','Logistic Regression','Random Forest']

select_option = st.selectbox('Choose Any Model:', options)

st.write('You selected: ', select_option)

if select_option == 'Logistic Regression':
    model = joblib.load('model_l.pkl')
elif select_option == 'Random Forest':
    model = joblib.load('model_rfc.pkl')

# Define a function to make predictions
def predict(model, X):
    # Preprocess your data as needed
    X_processed = X
    # Make predictions using your model
    y_pred = model.predict(X_processed)
    # Return the predicted labels
    return y_pred

def predict_landslide(input):
    prediction = model.predict(input)
    return prediction

# Define the layout of the app
st.title('Landslide Prediction Toolkit')
st.write('Enter the following details to predict landslide:')

lithology = st.text_input('ithology')
altitude = st.text_input('Altitude')
slope = st.text_input('Slope')
curvature = st.text_input('Total curvature')
aspect = st.text_input('Aspect')
distance_to_road = st.text_input('Distance to road')
distance_to_stream =st.text_input('Distance to stream')
slope_length = st.text_input('Slope length',)
twi = st.text_input("TWI")
sti = st.text_input('STI')

# Dictionary to assign input data
input_data = {
    'Lithology': lithology, 'Altitude': altitude, 'Slope': slope, 'Total curvature': curvature, 'Aspect': aspect,
    'Distance to road': distance_to_road, "Distance to stream": distance_to_stream, "Slope length": slope_length,
    "TWI": twi, 'STI': sti
    }

# Dictionary to Df
input_df = pd.DataFrame(input_data, index=[0])

input_df = input_df[['Lithology', 'Altitude', 'Slope', 'Total curvature', 'Aspect', 'Distance to road','Distance to stream', 'Slope length', 'TWI', "STI"  ]]

if st.button('Predict'):
    prediction = predict_landslide(input_df)
    if prediction == 1:
        st.write('Landslide')
    else:
        st.write('Landslide will not happen.')

