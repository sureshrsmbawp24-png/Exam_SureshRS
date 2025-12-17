import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. SETUP: Define Your X Variable Here ---
# Replace the list below with your actual column names from your X variable.
# Example: feature_names = ['Age', 'Salary', 'Credit_Score', 'Spending_Score']
feature_names = [
    'Feature 1', 
    'Feature 2', 
    'Feature 3', 
    'Feature 4'
]

# --- 2. Load the Model ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('log_reg_model_cv.joblib')
        return model
    except FileNotFoundError:
        st.error("Error: 'log_reg_model_cv.joblib' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- 3. App Layout ---
st.title("Dynamic Logistic Regression Deployment")
st.markdown("Enter values for the features below to generate a prediction.")
st.write("---")

# --- 4. Dynamic User Inputs ---
# This dictionary will store the user's answers
user_inputs = {}

# Create a grid layout (2 columns per row) for better visibility
cols = st.columns(2)

# Loop through your feature_names list and create an input field for each
for i, feature in enumerate(feature_names):
    # Select which column to place the field in (left or right)
    col = cols[i % 2]
    
    with col:
        # We assume numerical input for Logistic Regression. 
        # You can change value=0.0 to a typical average value for better UX.
        user_inputs[feature] = st.number_input(f"{feature}", value=0.0)

# --- 5. Prediction Logic ---
if st.button("Predict Result", type="primary"):
    if model:
        # Convert dictionary to DataFrame (ensures correct column order)
        input_df = pd.DataFrame([user_inputs])
        
        try:
            # Make Prediction
            prediction = model.predict(input_df)[0]
            
            # Display Result
            st.success(f"**Predicted Class:** {prediction}")
            
            # Display Probability (if available)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)
                probability = np.max(proba) * 100
                st.write(f"**Confidence:** {probability:.2f}%")
                
                # Simple bar chart of probabilities
                classes = model.classes_
                proba_df = pd.DataFrame(proba.T, columns=["Probability"], index=classes)
                st.bar_chart(proba_df)
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Tip: Ensure your 'feature_names' list matches exactly what the model was trained on.")