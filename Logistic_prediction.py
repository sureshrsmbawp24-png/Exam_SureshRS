import streamlit as st
import joblib
import pandas as pd

# --- 1. Load the Model ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('log_reg_model_cv.joblib')
        return model
    except FileNotFoundError:
        st.error("Error: 'log_reg_model_cv.joblib' not found.")
        return None

model = load_model()

# --- 2. App Layout ---
st.title("Logistic Regression Deployment")
st.markdown("Predict outcomes using your trained model.")

# --- 3. Sidebar: Auto-detect Feature Names ---
st.sidebar.header("Configuration")
st.sidebar.info("Upload a sample CSV (e.g., your X_train or dataset) so the app knows your column names.")

uploaded_file = st.sidebar.file_uploader("Upload CSV with correct columns", type=["csv"])

# Default fallback if no file is uploaded
feature_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']

if uploaded_file is not None:
    # Read only the first few rows to get column names
    df_sample = pd.read_csv(uploaded_file, nrows=5)
    
    # If the dataset has a target/label column (e.g., 'Target'), remove it!
    # UNCOMMENT the line below and replace 'Target' with your actual label name if needed:
    # df_sample = df_sample.drop(columns=['Target'], errors='ignore')
    
    feature_names = df_sample.columns.tolist()
    st.sidebar.success(f"✅ Detected {len(feature_names)} features!")

st.write("---")

# --- 4. Dynamic Input Fields ---
st.subheader("Enter Input Values")

# Dictionary to capture user inputs
user_inputs = {}

# Create a layout that adapts to the number of features
# If many features, use 3 columns; otherwise 2
cols_per_row = 3 if len(feature_names) > 6 else 2
cols = st.columns(cols_per_row)

for i, feature in enumerate(feature_names):
    col = cols[i % cols_per_row]
    with col:
        # Create a number input for each feature name
        user_inputs[feature] = st.number_input(f"{feature}", value=0.0)

# --- 5. Prediction Logic ---
if st.button("Predict Result", type="primary"):
    if model:
        # 1. Prepare data exactly as the model expects
        input_df = pd.DataFrame([user_inputs])
        
        # 2. Ensure columns are in the exact order the model expects
        # (This is why uploading the CSV is safer—it keeps the order correct)
        input_df = input_df[feature_names]
        
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"**Prediction:** {prediction}")
            
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)
                st.info(f"**Confidence:** {proba.max()*100:.2f}%")
        except Exception as e:
            st.error(f"Error: {e}")
            st.warning("Check if the uploaded CSV matches the columns the model was trained on.")