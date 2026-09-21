
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('logi.sav')

st.title('Delivery Delay Prediction App')
st.write('Enter the features below to predict if a delivery will be delayed.')

# Input fields for the features
# The order of features should match X.columns from the training data
features = [
    'Delivery_Distance',
    'Traffic_Congestion',
    'Weather_Condition',
    'Delivery_Slot',
    'Driver_Experience',
    'Num_Stops',
    'Vehicle_Age',
    'Road_Condition_Score',
    'Package_Weight',
    'Fuel_Efficiency',
    'Warehouse_Processing_Time'
]

input_data = {}
# You can customize min_value, max_value, and default values based on your data's characteristics
for feature in features:
    if feature in ['Traffic_Congestion', 'Weather_Condition', 'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age', 'Road_Condition_Score', 'Warehouse_Processing_Time']:
        input_data[feature] = st.slider(f'Select {feature.replace("_", " ")}', min_value=0, max_value=20, value=5)
    else: # Numerical features like Delivery_Distance, Package_Weight, Fuel_Efficiency
        input_data[feature] = st.slider(f'Select {feature.replace("_", " ")}', min_value=0.0, max_value=100.0, value=25.0, step=0.1)

# Create a DataFrame from the input data
input_df = pd.DataFrame([input_data])

if st.button('Predict Delivery Delay'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error('The delivery is likely to be delayed!')
    else:
        st.success('The delivery is likely to be on time.')
    
    st.write(f'Probability of On-Time (0): {prediction_proba[0][0]:.2f}')
    st.write(f'Probability of Delay (1): {prediction_proba[0][1]:.2f}')
