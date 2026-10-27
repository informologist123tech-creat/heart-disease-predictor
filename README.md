# ❤️ Heart Disease Predictor

A machine learning web application built with **Python, Streamlit, Pandas, Plotly, and Scikit-learn** to demonstrate heart disease prediction using a trained Random Forest classifier.

## 📌 Project Overview

This project provides a simple web interface where users can enter health-related input features and receive a model-generated prediction. It also supports bulk predictions by uploading a CSV file and provides a comparison of machine learning model accuracy.

## ✨ Features

- **Single Prediction:** Enter individual feature values through an interactive interface.
- **Bulk Prediction:** Upload a CSV file to generate predictions for multiple rows.
- **Download Results:** Export bulk prediction results as a CSV file.
- **Model Information:** View a chart comparing reported test accuracy for different models.
- **Interactive Interface:** Built using Streamlit and Plotly.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Plotly
- Jupyter Notebook

## 🤖 Machine Learning Models

The project explored the following models:

| Model | Reported Test Accuracy |
|---|---:|
| Logistic Regression | 85.87% |
| Decision Tree | 80.98% |
| Random Forest | 88.59% |

These are the results recorded during development. They are not a guarantee of performance on new data or evidence of clinical validity.

## 📊 Input Features

The model uses 11 input features:

Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, Resting ECG, Maximum Heart Rate, Exercise-Induced Angina, Oldpeak, and ST Slope.

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/informologist123tech-creat/heart-disease-predictor.git
cd heart-disease-predictor
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

## 📁 Project Structure

```text
heart-disease-predictor/
├── app.py
├── random_forest.pkl
├── requirements.txt
├── README.md
└── HEART_DISEASES_PREDICTOR.ipynb
```

The notebook is optional and should be included only if you want to share the model-training workflow.

## ⚠️ Disclaimer

This application is an educational machine learning project. Its predictions are not medical diagnoses and must not be used to make medical decisions. Consult a qualified healthcare professional for medical assessment.

## 👨‍💻 Author

**Okasha Amir**

Computer Science Student | Machine Learning Enthusiast

