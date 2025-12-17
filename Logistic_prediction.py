import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. SETUP: Feature Names Extracted from your CSV ---
# These are the columns used as features (X) for your model.
# I've excluded 'Late_Delivery_Status' assuming it is the target (Y).
FEATURE_NAMES = [
    'Supplier_Quality_Score',
    'Cost_Per_Unit',
    'Lead_Time_Days',
    'Route_ID_Asia-US',
    'Route_ID_Domestic',
    'Route_ID_Europe-US',
    'Transport_Mode_Sea',
    'Transport_Mode_Truck',
    'Transport_Mode_Unknown'
]

# --- 2. Load the Model ---
@st.cache_resource
def load_model():
    """Loads the trained model from the local directory."""
    try:
        model = joblib.load('log_reg_model_cv.joblib')
        return model
    except FileNotFoundError:
        st.error("Error: 'log_reg_model_cv.joblib' not found. Please ensure it's in the same folder.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading the model: {e}")
        return None

model = load_model()

# --- 3. App Layout ---
st.title("Logitech Supply Chain Prediction")
st.markdown("Predict **Late Delivery Status** using the Logistic Regression model.")
st.write("---")

# --- 4. Dynamic User Inputs ---
st.header("Input Supply Chain Metrics")

# Dictionary to capture user inputs
user_inputs = {}

# Use 3 columns for a clean layout
cols = st.columns(3)

# Loop through your feature_names list and create an input field for each
for i, feature in enumerate(FEATURE_NAMES):
    col = cols[i % 3] # Cycle through the three columns
    
    with col:
        # For boolean/one-hot-encoded columns (starting with 'Route' or 'Transport'), use a checkbox
        if feature.startswith('Route_ID_') or feature.startswith('Transport_Mode_'):
            # Checkbox returns True/False, convert to 1/0 for model input
            is_active = st.checkbox(feature, value=False)
            user_inputs[feature] = 1.0 if is_active else 0.0
        else:
            # For numerical columns, use a number input
            user_inputs[feature] = st.number_input(f"{feature}", value=0.5, step=0.01)

# --- 5. Prediction Logic ---
if st.button("Predict Late Delivery Status", type="primary"):
    if model is not None:
        # Create DataFrame with the user's inputs
        input_df = pd.DataFrame([user_inputs])
        
        # IMPORTANT: Reorder columns to ensure they match the model's training order
        input_df = input_df[FEATURE_NAMES]
        
        try:
            # Make the prediction
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)
            
            # --- 6. Display Results ---
            st.success("Prediction Complete")
            
            # Since Late_Delivery_Status is likely 0 or 1, display meaningful message
            status = "LATE DELIVERY (Status = 1)" if prediction == 1 else "ON-TIME DELIVERY (Status = 0)"
            
            st.metric(label="Predicted Delivery Status", value=status)
            
            # Display Probability (Confidence)
            confidence = np.max(proba) * 100
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            # Visualize probabilities
            classes = model.classes_
            proba_df = pd.DataFrame(proba.T, columns=["Probability"], index=[f"Class {c}" for c in classes])
            st.bar_chart(proba_df)

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.warning("Ensure the values are in the correct format for the model.")
    else:
        st.error("Model not loaded.")

The video, [How to Read Excel Files with Python (Pandas Tutorial)](https://www.youtube.com/watch?v=P6HCyxSyFpY), provides a good fundamental understanding of how to import data from files like your CSV using Pandas, which is essential for defining the features in this Streamlit app.


http://googleusercontent.com/youtube_content/0