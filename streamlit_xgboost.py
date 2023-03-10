import streamlit as st
from  datetime import date, time, datetime
import joblib
import pandas as pd
import category_encoders as ce
# Min Max Scaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

from sklearn.ensemble import AdaBoostClassifier 
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

import os
from sklearn.metrics import classification_report, confusion_matrix, \
accuracy_score, recall_score, precision_score, f1_score
import category_encoders as ce

colunas = ['region_category', 'membership_category', 'joining_date',
       'joined_through_referral', 'preferred_offer_types', 'last_visit_time',
       'days_since_last_login', 'avg_transaction_value', 'points_in_wallet',
       'used_special_discount', 'offer_application_preference',
       'past_complaint', 'feedback']

st.write("Churn Prediction - Prediction using XGBoost")
st.write("We also used Lasso's feature selection.")


# getting user inputgender = col1.selectbox("Enter your gender",["Male", "Female"])


region_category = st.selectbox("Enter the region",['City', 'Town', 'Village'])

membership_category = st.selectbox("Enter the membership",
                      ['No Membership', 'Gold Membership', 'Silver Membership',
       'Basic Membership', 'Premium Membership', 'Platinum Membership'])

joining_date = st.date_input('Joining Date', date(2019, 7, 6))


joined_through_referral = st.selectbox('Joined through referral?',["Yes","No"])

preferred_offer_types = st.selectbox('Preferred offer? ', ['Without Offers', 'Gift Vouchers/Coupons',
       'Credit/Debit Card Offers'])

#last_visit_time  = st.date_input('Last Visit Date', value='datetime.time')

last_visit_time = st.time_input('Last Visit Time', time(8, 45))


days_since_last_login = st.number_input('Days since last login: ')

points_in_wallet = st.number_input('Points in wallet: ')

used_special_discount = st.selectbox('Special discounts? ', ['Yes', 'No'])

offer_application_preference = st.selectbox('Offer Application Preference? ', ['Yes', 'No'])

past_complaint = st.selectbox('Past Complaint? ', ['Yes', 'No'])

avg_transaction_value = st.number_input('Average transactional value: ')

feedback = st.selectbox('Feedback: ', ['Poor Product Quality', 'Too many ads', 'Poor Customer Service',
       'No reason specified', 'Poor Website', 'Quality Customer Care',
       'User Friendly Website', 'Reasonable Price',
       'Products always in Stock'])



df_pred = pd.DataFrame([[region_category, membership_category, joining_date,
       joined_through_referral, preferred_offer_types, last_visit_time,
       days_since_last_login, avg_transaction_value, points_in_wallet,
       used_special_discount, offer_application_preference,
       past_complaint, feedback]], columns = colunas )

#df_pred['joining_date'] = [date.fromisoformat(x) for x in df_pred['joining_date']]
#df_pred['last_visit_time'] =  \
#[datetime.strptime(x, '%H:%M:%S').time() for x in df_pred['last_visit_time']]


numerical = df_pred._get_numeric_data().columns
categorical = list(set(df_pred.columns) - set(numerical))
encoder = ce.OrdinalEncoder(categorical)
df_pred = encoder.fit_transform(df_pred)

names = df_pred.columns
indexes = df_pred.index
sc = MinMaxScaler((0, 1))
data = df_pred
df_pred = sc.fit_transform(df_pred)
data_scaled = pd.DataFrame(df_pred, columns=names, index=indexes)

model = joblib.load('xgboost_model.pkl')
prediction = model.predict(df_pred)


if st.button('Churn Prediction!'):
   
    st.write('Your churn rate is: ', prediction)
    