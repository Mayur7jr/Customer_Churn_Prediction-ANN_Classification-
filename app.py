import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

# Load model & preprocessors
model = load_model('model.h5')

with open('onehot_enocoder_geo.pkl', 'rb') as file:
    onehotencoder = pickle.load(file)

with open('label_encoded_gender.pkl', 'rb') as file:
    labelencoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# App title
st.markdown("<h1 style='text-align: center;'>📊 Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict whether a customer is likely to leave the bank</p>", unsafe_allow_html=True)
st.divider()

# ------------------ User Inputs ------------------
st.subheader("🧾 Customer Information")

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('🌍 Geography', onehotencoder.categories_[0])
    gender = st.selectbox('👤 Gender', labelencoder.classes_)
    age = st.slider('🎂 Age', 18, 92)
    tenure = st.slider('📅 Tenure (Years)', 0, 10)
    num_of_products = st.slider('📦 Number of Products', 1, 4)

with col2:
    credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=900, step=1)
    balance = st.number_input('💰 Account Balance')
    estimated_salary = st.number_input('💼 Estimated Salary')
    has_cr_card = st.selectbox('💳 Has Credit Card', ['No', 'Yes'])
    is_active_member = st.selectbox('⚡ Active Member', ['No', 'Yes'])

# Convert Yes/No to 0/1
has_cr_card = 1 if has_cr_card == 'Yes' else 0
is_active_member = 1 if is_active_member == 'Yes' else 0

st.divider()

# ------------------ Prediction ------------------
if st.button("🔮 Predict Churn", use_container_width=True):

    # Prepare input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [labelencoder.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode Geography
    geo_encoded = onehotencoder.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehotencoder.get_feature_names_out(['Geography'])
    )

    # Combine data
    input_data = pd.concat([input_data, geo_encoded_df], axis=1)

    # Scale
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)
    churn_probability = prediction[0][0]

    st.divider()
    st.subheader("📌 Prediction Result")

    # Display result nicely
    st.metric(
        label="Churn Probability",
        value=f"{churn_probability * 100:.2f}%"
    )

    if churn_probability > 0.5:
        st.error("🚨 The customer is **likely to churn**.")
    else:
        st.success("✅ The customer is **not likely to churn**.")
