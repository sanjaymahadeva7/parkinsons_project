import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler

# Load the saved model
model_file = 'random_forest_model.joblib'
rf_model = load(model_file)

# Function to preprocess data and make predictions
def make_predictions(df):
    # Drop the 'name' column if it exists
    if 'name' in df.columns:
        df = df.drop(columns=['name'])

    # Ensure 'status' column is present and separate it
    if 'status' in df.columns:
        X = df.drop(columns=['status'])
        y = df['status']
    else:
        X = df
        y = None  # No actual status values to compare with

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Make predictions
    y_pred = rf_model.predict(X)

    # Create a DataFrame with the actual and predicted status
    if y is not None:
        results_df = pd.DataFrame({
            'Actual': y,
            'Predicted': y_pred
        })
    else:
        results_df = pd.DataFrame({
            'Predicted': y_pred
        })

    return results_df

# Streamlit UI
st.title('Random Forest Model Deployment')

# Add custom CSS for background image
background_image_url = 'https://dividat.com/assets/containers/main/us/parkinsons.jpg/6fd0d37dce18d335aa7d33d8369928b0.jpg'  # Replace with your image URL
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)

    # Display the uploaded dataset
    st.write("Uploaded Dataset:")
    st.write(df)

    # Make and display predictions
    results_df = make_predictions(df)
    st.write("Predictions:")
    st.write(results_df)


# Define the input fields for all features
features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 
            'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
            'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 
            'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']

input_data = {}

# Collect user inputs for each feature
for feature in features:
    input_data[feature] = st.number_input(f'Enter {feature}', value=0.0)

# Convert the input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Standardize the features
scaler = StandardScaler()
input_df_scaled = scaler.fit_transform(input_df)

def map_prediction(prediction):
    return "Healthy" if prediction == 0 else "Parkinson's"

# Make predictions
if st.button('Predict'):
    y_pred = rf_model.predict(input_df_scaled)
    prediction_status = map_prediction(y_pred[0])
    st.write(f'Predicted Status: {prediction_status}')