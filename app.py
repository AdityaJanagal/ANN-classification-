import tensorflow as tf
import pickle
import pandas as pd 
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder

model = tf.keras.models.load_model('model.h5')

with open('Label_encoder_gender.pkl','rb') as file:
    label_encoder_gender= pickle.load(file)
    
with open('One_hot_encoder_geo.pkl','rb') as file:
    One_hot_encoder_geo= pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler= pickle.load(file)


##Streamlit app

st.title("🏦 Customer Churn Prediction")
st.write("Enter customer details to predict whether the customer is likely to churn.")

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=650
)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

num_of_products = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=1
)

has_cr_card = st.selectbox(
    "Has Credit Card?",
    ["Yes", "No"]
)

is_active_member = st.selectbox(
    "Is Active Member?",
    ["Yes", "No"]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

if st.button("Predict Churn"):

    # Convert Yes/No to 1/0
    has_cr_card_value = 1 if has_cr_card == "Yes" else 0

    is_active_member_value = 1 if is_active_member == "Yes" else 0


    # Create dataframe
    input_data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": has_cr_card_value,
        "IsActiveMember": is_active_member_value,
        "EstimatedSalary": estimated_salary
    }

    input_data_df = pd.DataFrame([input_data])


    # =========================
    # ONE-HOT ENCODE GEOGRAPHY
    # =========================

    encoder_geo = One_hot_encoder_geo.transform(
        input_data_df[["Geography"]]
    )

    encoded_df = pd.DataFrame(
        encoder_geo,
        columns=One_hot_encoder_geo.get_feature_names_out(
            ["Geography"]
        ),
        index=input_data_df.index
    )


    # =========================
    # LABEL ENCODE GENDER
    # =========================

    input_data_df["Gender"] = label_encoder_gender.transform(
        input_data_df["Gender"]
    )


    # Remove original Geography
    input_data_df = input_data_df.drop(
        columns=["Geography"]
    )


    # =========================
    # COMBINE DATA
    # =========================

    input_data_df = pd.concat(
        [encoded_df, input_data_df],
        axis=1
    )


    # =========================
    # MATCH SCALER COLUMN ORDER
    # =========================

    input_data_df = input_data_df[
        scaler.feature_names_in_
    ]


    # =========================
    # SCALE DATA
    # =========================

    scaled_input = scaler.transform(
        input_data_df
    )


    # =========================
    # PREDICT
    # =========================

    prediction = model.predict(
        scaled_input,
        
    )

    probability = prediction[0][0]

    percentage = probability * 100


    # =========================
    # DISPLAY RESULT
    # =========================

    st.subheader("Prediction Result")

    st.write(
        f"Churn Probability: **{percentage:.2f}%**"
    )


    if probability >= 0.5:

        st.error("⚠️ Customer is likely to CHURN")

    else:

        st.success("✅ Customer is likely to STAY")
