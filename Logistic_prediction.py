import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Load the Model ---
@st.cache_resource
def load_model():
    """
    Loads the trained model from the local directory.
    Using @st.cache_resource prevents reloading the model on every interaction.
    """
    try:
        # Load the specific joblib file provided
        model = joblib.load('log_reg_model_cv.joblib')
        return model
    except FileNotFoundError:
        st.error("Error: 'log_reg_model_cv.joblib' not found. Please upload it to the same directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading the model: {e}")
        return None

model = load_model()

# --- 2. App Layout ---
st.title("Logistic Regression Model Deployment")
st.markdown("""
This application predicts outcomes using your pre-trained **Logistic Regression** model.
Enter the feature values below to get a prediction.
""")
st.write("---")

# --- 3. User Input Section ---
st.header("Input Features")
st.info("Update the fields below to match the features your model was trained on.")

# Create columns for a cleaner layout
col1, col2 = st.columns(2)

# === CUSTOMIZE THESE INPUTS ===
# Replace 'Feature 1', 'Feature 2' etc. with your actual dataset column names (e.g., 'Age', 'Salary')
with col1:
    f1 = st.number_input("Feature 1", value=0.0)
    f2 = st.number_input("Feature 2", value=0.0)

with col2:
    f3 = st.number_input("Feature 3", value=0.0)
    f4 = st.number_input("Feature 4", value=0.0)
# Add more input fields here if your model has more than 4 features

# --- 4. Prediction Logic ---
if st.button("Predict Result", type="primary"):
    if model is not None:
        # Organize inputs into a DataFrame
        # Ensure the column names and ORDER match exactly what the model expects
        input_data = pd.DataFrame([[f1, f2, f3, f4]], 
                                  columns=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'])
        
        try:
            # Make the prediction
            prediction = model.predict(input_data)[0]
            
            # Get probability (confidence) if supported
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_data)
                probability = np.max(proba) * 100
            else:
                probability = None

            # --- 5. Display Results ---
            st.success("Prediction Complete")
            
            st.metric(label="Predicted Class", value=str(prediction))
            
            if probability:
                st.write(f"**Confidence:** {probability:.2f}%")
                st.bar_chart(pd.DataFrame(proba.T, columns=["Probability"], index=model.classes_))

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.warning("Hint: Check that the number of input fields matches the number of features the model expects.")
    else:
        st.error("Model not loaded.")

# Optional: View Model Parameters (for debugging)
with st.expander("See Model Parameters"):
    if model:
        st.json(model.get_params())