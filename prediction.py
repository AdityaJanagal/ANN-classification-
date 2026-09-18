from tensorflow.keras.models import load_model
import pickle
import pandas as pd 
import numpy as np

model = load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender= pickle.load(file)
    
with open('One_hot_encoder_geo.pkl','rb') as file:
    One_hot_encoder_geo= pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler= pickle.load(file)

input_data = {
    "RowNumber": 5000,
    "CustomerId": 15738492,
    "Surname": "Smith",
    "CreditScore": 450,
    "Geography": "Germany",
    "Gender": "Female",
    "Age": 55,
    "Tenure": 2,
    "Balance": 150000.00,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 50000.00,
    "Exited": 0
}

input_data_df = pd.DataFrame([input_data])
input_data_df = input_data_df.drop(columns=[ 'RowNumber','CustomerId','Surname','Exited'])

encoder_geo = One_hot_encoder_geo.transform(input_data_df[['Geography']].astype(str))
encoded_df = pd.DataFrame(encoder_geo,columns=One_hot_encoder_geo.get_feature_names_out(['Geography']))

label_gender = label_encoder_gender.transform(input_data_df['Gender'])
input_data_df['Gender'] = label_gender

input_data_df=input_data_df.drop(columns = ['Geography'])
input_data_df = pd.concat(
    [
    encoded_df,input_data_df],
    axis=1
)

scaled_input = scaler.transform(input_data_df)
prediction = model.predict(scaled_input)

if prediction[0][0] > 0.5:
    print("Customer will likely churn")
else:
    print("Customer will likely stay")


