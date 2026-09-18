
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import streamlit as st


# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model("RegressionModel.h5")


# =========================
# LOAD ENCODERS
# =========================

with open("Label_encoder.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("OneHot_Encoder.pkl", "rb") as file:
    One_hot_encoder_geo = pickle.load(file)


# =========================
# LOAD SCALER
# =========================

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# =========================
# STREAMLIT APP
# =========================

st.title("💰 Estimated Salary Prediction")

st.write(
    "Enter customer details to predict the estimated salary."
)


# =========================
# USER INPUTS
# =========================

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

is_exited = st.selectbox(
    "Exited" , 
    ['Yes','No']
)

# =========================
# PREDICTION
# =========================

if st.button("Predict Salary"):

    # Convert Yes/No to 1/0

    has_cr_card_value = (
        1 if has_cr_card == "Yes" else 0
    )

    is_active_member_value = (
        1 if is_active_member == "Yes" else 0
    )
    is_exited = (
        1 if is_exited == "Yes" else 0
    )



    # =========================
    # CREATE DATAFRAME
    # =========================

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

        "Exited" : is_exited




    }


    input_data_df = pd.DataFrame(
        [input_data]
    )


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

    input_data_df["Gender"] = (
        label_encoder_gender.transform(
            input_data_df["Gender"]
        )
    )


    # =========================
    # REMOVE ORIGINAL GEOGRAPHY
    # =========================

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
    # MATCH TRAINING COLUMNS
    # =========================

    input_data_df = input_data_df[
        scaler.feature_names_in_
    ]


    # =========================
    # SCALE INPUT
    # =========================

    scaled_input = scaler.transform(
        input_data_df
    )


    # =========================
    # PREDICT SALARY
    # =========================

    prediction = model.predict(
        scaled_input,
        verbose=0
    )



    predicted_salary = prediction[0][0]


    # =========================
    # DISPLAY RESULT
    # =========================

    st.subheader("Prediction Result")

    st.success(
        f"💰 Estimated Salary: ₹{predicted_salary:,.2f}"
    )
