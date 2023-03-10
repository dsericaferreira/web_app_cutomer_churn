import streamlit as st
import datetime
import joblib
import pandas as pd
import category_encoders as ce


st.write("Churn Prediction - Prediction using Random Forest")

gender = st.selectbox("Enter the gender",["Male", "Female"])

# getting user inputgender = col1.selectbox("Enter your gender",["Male", "Female"])

age = st.number_input("Enter the age")

region_category = st.selectbox("Enter the region",['City', 'Town', 'Village'])

membership_category = st.selectbox("Enter the membership",
                      ['No Membership', 'Gold Membership', 'Silver Membership',
       'Basic Membership', 'Premium Membership', 'Platinum Membership'])

#joining_date = st.date_input('Joining Date', value='datetime.date')

joined_through_referral = st.selectbox('Joined through referral?',["Yes","No"])

preferred_offer_types = st.selectbox('Preferred offer? ', ['Without Offers', 'Gift Vouchers/Coupons',
       'Credit/Debit Card Offers'])

days_since_last_login = st.number_input('Days since last login: ')

avg_time_spent = st.number_input('Average time spent: ')

avg_transaction_value = st.number_input('Average transactional value: ')

feedback = st.selectbox('Feedback: ', ['Poor Product Quality', 'Too many ads', 'Poor Customer Service',
       'No reason specified', 'Poor Website', 'Quality Customer Care',
       'User Friendly Website', 'Reasonable Price',
       'Products always in Stock'])



df_pred = pd.DataFrame([[age, gender, region_category, membership_category,
                          joined_through_referral, preferred_offer_types,
       days_since_last_login, avg_time_spent, avg_transaction_value,
       feedback]], columns = ['age', 'gender', 'region_category', 'membership_category',
       'joined_through_referral', 'preferred_offer_types',
       'days_since_last_login', 'avg_time_spent', 'avg_transaction_value',
       'feedback'])

numerical = df_pred._get_numeric_data().columns
categorical = list(set(df_pred.columns) - set(numerical))
encoder = ce.OrdinalEncoder(categorical)
df_pred = encoder.fit_transform(df_pred)
model = joblib.load('random_model.pkl')
prediction = model.predict(df_pred)

if st.button('Churn Prediction!'):
   
    st.write('Your churn rate is: ', prediction[0])