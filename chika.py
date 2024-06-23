# to deploy it we type, streamlit run chika.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sklearn

def load_model(model_path='gb_grid_search_joblib.pkl'):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")


def predict_rating(model, input_features):
    input_array = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

def main():
    try:
        rf_model = load_model('gb_grid_search_joblib.pkl') 
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        return

    st.title("Predict the overall rating of a football player:")
    st.write('Select the features below')
    

    overall = st.number_input('Overall Rating', min_value=0, max_value=100, value=50, key='overall')
    movement_reactions = st.number_input('Movement Reactions', min_value=0, max_value=100, value=50, key='movement_reactions')
    mentality_composure = st.number_input('Mentality Composure', min_value=0, max_value=100, value=50, key='mentality_composure')
    potential = st.number_input('Potential', min_value=0, max_value=100, value=50, key='potential')
    wage_eur = st.number_input('Wage (EUR)', min_value=0, value=0, key='wage_eur')
    rcm = st.number_input('RCM', min_value=0, max_value=100, value=50, key='rcm')
    cm = st.number_input('CM', min_value=0, max_value=100, value=50, key='cm')

    if st.button('Overall Rating'):
        input_features = [
            overall, movement_reactions, mentality_composure, potential, wage_eur, rcm, cm
        ]
        try:
            rf_prediction = predict_rating(rf_model, input_features)
            st.success(f'Random Forest predicted overall rating: {rf_prediction:.2f}')
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()