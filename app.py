import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import base64
from pathlib import Path
from sklearn.utils.validation import check_is_fitted


MODEL_PATH = Path(__file__).resolve().parent / "random_forest.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    check_is_fitted(model)
    return model

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background-color: #0B1220;
    color: #F1F5F9;
}

section[data-testid="stSidebar"] {
    background-color: #111B2E;
}

h1, h2, h3 {
    color: #F1F5F9;
}

div.stButton > button {
    background-color: #0D9488;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1rem;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #0F766E;
    color: white;
    border: 1px solid #5EEAD4;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    padding: 24px;
    border-radius: 16px;
    background: linear-gradient(120deg, #14243B, #0F766E);
    margin-bottom: 20px;
">
    <h1>❤️ Heart Disease Prediction System</h1>
    <p style="font-size: 17px; color: #E2E8F0;">
        Machine Learning · Health Data Analysis · Model Insights
    </p>
    <p style="color: #E2E8F0;">
        Built with ❤️ by OKASHA AMIR
    </p>
</div>
""", unsafe_allow_html=True)





tab_1,tab_2,tab_3 = st.tabs(['Predict' , 'Bulk Predict' , 'Model Information'])


with tab_1:
    

    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
    )
    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)", min_value=0, max_value=300
    )
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0)
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"]
    )
    resting_ecg = st.selectbox(
        "Resting ECG Results",
        ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
    )
    max_hr = st.number_input(
        "Maximum Heart Rate Achieved", min_value=60, max_value=202
    )
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input(
        "Oldpeak (ST Depression)", min_value=0.0, max_value=10.0
    )
    st_slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        ["Upsloping", "Flat", "Downsloping"]
    )

    # Convert categorical inputs to numeric
    sex_value = 0 if sex == "Male" else 1

    chest_pain_value = [
        "Atypical Angina",
        "Non-Anginal Pain",
        "Asymptomatic",
        "Typical Angina"
    ].index(chest_pain)

    fasting_bs_value = 1 if fasting_bs == "> 120 mg/dl" else 0

    resting_ecg_value = [
        "Normal",
        "ST-T Wave Abnormality",
        "Left Ventricular Hypertrophy"
    ].index(resting_ecg)

    exercise_angina_value = 1 if exercise_angina == "Yes" else 0

    st_slope_value = [
        "Upsloping", "Flat", "Downsloping"
    ].index(st_slope)

    # Create the input DataFrame
    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex_value],
        "ChestPainType": [chest_pain_value],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs_value],
        "RestingECG": [resting_ecg_value],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina_value],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope_value]
    })

    # The Predict button now belongs ONLY to Tab 1
    if st.button("Predict", type="primary"):
        try:
            model = load_model()
            prediction = model.predict(input_data)[0]

            st.subheader("Heart Disease Prediction")

            if prediction == 0:
                st.success("Model prediction: No heart disease")
            else:
                st.warning("Model prediction: Heart disease")

            st.caption(
                "This is a machine-learning prediction, not a medical "
                "diagnosis. Consult a qualified healthcare professional "
                "for medical assessment."
            )

        except Exception as error:
            st.error(f"Prediction error: {error}")




# TAB 2: CSV FILE PREDICTIONS
with tab_2:
    st.title("Upload CSV File")

    st.subheader("Instructions before uploading")

    st.info("""
    1. Upload a CSV file containing the 11 required features.
    2. Do not include missing (NaN) values.
    3. Use the exact column names shown below.
    4. Feature values must follow the numerical encoding used
       when training the Random Forest model.
    5. Extra columns are allowed, but only the 11 model features
       will be used for prediction.
    """)

    # Required features in the correct order
    expected_columns = [
        "Age",
        "Sex",
        "ChestPainType",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "RestingECG",
        "MaxHR",
        "ExerciseAngina",
        "Oldpeak",
        "ST_Slope"
    ]

    st.caption(
        "Required columns: " + ", ".join(expected_columns)
    )

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        key="csv_prediction_upload"
    )

    if uploaded_file is not None:
        try:
            # Read the uploaded CSV
            input_data = pd.read_csv(uploaded_file)

            # Check required columns
            missing_columns = [
                col for col in expected_columns
                if col not in input_data.columns
            ]

            if missing_columns:
                st.error(
                    "Your CSV is missing these columns: "
                    + ", ".join(missing_columns)
                )

            elif input_data.empty:
                st.error("The uploaded CSV contains no data rows.")

            else:
                # Keep only the model features in training order
                features = input_data[expected_columns].copy()

                # Check for missing values
                if features.isnull().any().any():
                    st.error(
                        "Your CSV contains missing values. "
                        "Please remove or correct them and upload again."
                    )

                else:
                    # Ensure all feature values are numeric
                    for col in expected_columns:
                        features[col] = pd.to_numeric(
                            features[col],
                            errors="raise"
                        )

                    # Load the Random Forest model
                    model = load_model()

                    # Predict all rows at once
                    predictions = model.predict(features)

                    # Keep the original CSV columns and add predictions
                    result_data = input_data.copy()
                    result_data["Prediction_RF"] = predictions

                    st.success(
                        f"Predictions completed for "
                        f"{len(result_data)} rows."
                    )

                    st.subheader("Prediction Results")
                    st.dataframe(
                        result_data,
                        use_container_width=True
                    )

                    # Download the predictions as a CSV
                    csv_bytes = result_data.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Predictions CSV",
                        data=csv_bytes,
                        file_name="PredictedHeart.csv",
                        mime="text/csv"
                    )

                    st.caption(
                        "Predictions are generated by a machine-learning "
                        "model and are not medical diagnoses."
                    )

        except ValueError as error:
            st.error(
                "Invalid CSV data. Check that all 11 features "
                "contain valid numerical values."
            )
            st.caption(str(error))

        except Exception as error:
            st.error(f"Prediction error: {error}")

    else:
        st.info("Upload a CSV file to get predictions.")



with tab_3:

    import plotly.express as px
 

    st.title("Model Information")

   

    data = {
        "Decision Tree": 80.98,
        "Logistic Regression": 85.87,
        "Random Forest": 88.59,
    }

    df_models = pd.DataFrame(
        list(data.items()),
        columns=["Model", "Accuracy (%)"]
    )

    fig = px.bar(
        df_models,
        x="Model",
        y="Accuracy (%)",
        text="Accuracy (%)",
        title="Test Accuracy of Machine Learning Models",
        labels={
            "Model": "Machine Learning Model",
            "Accuracy (%)": "Accuracy (%)"
        }
    )

    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(
        yaxis_range=[0, 100],
        yaxis_title="Accuracy (%)",
        xaxis_title="Model"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Accuracy Details")
    st.dataframe(df_models, use_container_width=True)

    st.caption(
     
        "They are not a guarantee of future prediction performance."
    )








