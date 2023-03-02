import streamlit as st
import pandas as pd
import pickle

# Load the saved model pipeline from disk using pickling
with open('model_pipeline.pickle', 'rb') as f:
    model_pipeline = pickle.load(f)

# Load the saved feature names from disk using pickling
with open('feature_names.pickle', 'rb') as f:
    feature_names = pickle.load(f)

# Define the user interface
st.title('Iris Flower Species Predictor')

# Define input fields for the user interface
sepal_length = st.slider('Sepal length', min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.slider('Sepal width', min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.slider('Petal length', min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.slider('Petal width', min_value=0.0, max_value=10.0, step=0.1)

# Store the user input in a pandas DataFrame
input_data = pd.DataFrame({
    'sepal length (cm)': [sepal_length],
    'sepal width (cm)': [sepal_width],
    'petal length (cm)': [petal_length],
    'petal width (cm)': [petal_width]
})

# Use the saved model pipeline to make predictions on the user input
predicted_species = model_pipeline.predict(input_data)

# Map the predicted species codes to their names
species_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
predicted_species_name = species_names[predicted_species[0]]

# Display the predictions to the user
st.subheader('Prediction:')
st.write(f'The predicted species is {predicted_species_name}')

# Display the feature names to the user
st.subheader('Features:')
for feature in feature_names:
    st.write(f'- {feature}')
