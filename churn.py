## Imports:
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


dataset = pd.read_csv('customer_churn.csv')
dataset = dataset.dropna()
dataset = dataset.replace('?', 'No')
dataset = dataset.replace(-999, 0)
dataset = dataset.drop(['Unnamed: 0', 'avg_frequency_login_days','last_visit_time', 'Name', 'customer_id', 'security_no', 'referral_id', 'medium_of_operation'], axis=1)
x_data = dataset.drop(['churn_risk_score'], axis=1)
y_data = dataset['churn_risk_score']
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size = 0.3)
numerical = dataset._get_numeric_data().columns
categorical = list(set(dataset.columns) - set(numerical))
## Colunas categóricas serão transformadas em ordinais:
encoder = ce.OrdinalEncoder(categorical)
x_train = encoder.fit_transform(x_train)
x_test = encoder.transform(x_test)
rfc = RandomForestClassifier(n_estimators=115, max_features='log2')
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
 #Saving the model 
print('Your accuracy is: ', accuracy)
joblib.dump(rfc, 'model.pkl') 


