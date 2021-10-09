import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = pd.read_csv("/kaggle/input/credit-card-approval-prediction/application_record.csv")
crecord = pd.read_csv("/kaggle/input/credit-card-approval-prediction/credit_record.csv")

#pd.set_option('display.max_colwidth', 0)
labels = app.columns.values.tolist()
info = ['Customer ID', 'Gender of customer', 'Car ownership', 'Property ownership', 'Number of children', 
        'Annual income', 'Income category (Working/Pensioner)', "Education Level (Higher education/Secondary)", 
        'Marital status', "Type of House (Rented/With parents)", 'Birthday', 
        'Duration of employment', 'Mobile ownership', 'Work phone ownership', 'Phone ownership', 'Email', 'Type of Occupation', 'Family Size']
info_app = pd.DataFrame(list(zip(labels, info)), columns = ['Feature Name', 'Explanation'])

labels1 = crecord.columns.values.tolist()
labels1
info1 = ['Customer ID', 'Monthly Balance', 'Status of Monthly Payment']
info_cred = pd.DataFrame(list(zip(labels1, info1)), columns = ['Feature Name', 'Explanation'])

app = app.drop_duplicates('ID', keep='last')
app.drop('OCCUPATION_TYPE', axis=1, inplace=True)
le = LabelEncoder()
for x in app:
    if app[x].dtypes=='object':
        app[x] = le.fit_transform(app[x])
# FOR CNT_CHILDREN COLUMN
q_hi = app['CNT_CHILDREN'].quantile(0.999)
q_low = app['CNT_CHILDREN'].quantile(0.001)
app = app[(app['CNT_CHILDREN'] > q_low) & (app['CNT_CHILDREN'] < q_hi)]
# FOR AMT_INCOME_TOTAL COLUMN
q_hi = app['AMT_INCOME_TOTAL'].quantile(0.999)
q_low = app['AMT_INCOME_TOTAL'].quantile(0.001)
app = app[(app['AMT_INCOME_TOTAL'] > q_low) & (app['AMT_INCOME_TOTAL'] < q_hi)]
#FOR CNT_FAM_MEMBERS COLUMN
q_hi = app['CNT_FAM_MEMBERS'].quantile(0.999)
q_low = app['CNT_FAM_MEMBERS'].quantile(0.001)
app = app[(app['CNT_FAM_MEMBERS'] > q_low) & (app['CNT_FAM_MEMBERS'] < q_hi)]

crecord['Months from today'] = crecord['MONTHS_BALANCE']*-1
crecord = crecord.sort_values(['ID','Months from today'], ascending=True)
crecord['STATUS'].replace({'C': 0, 'X' : 0}, inplace=True)
crecord['STATUS'] = crecord['STATUS'].astype('int')
crecord['STATUS'] = crecord['STATUS'].apply(lambda x:1 if x >= 2 else 0)

crecord_des = crecord.describe()
crecord_des.to_csv('cred.csv', index = True)
crecord_des

crecordgb = crecord.groupby('ID').agg(max).reset_index()
df = app.join(crecordgb.set_index('ID'), on='ID', how='inner')
df.drop(['Months from today', 'MONTHS_BALANCE', 'FLAG_MOBIL'], axis = 1, inplace = True)
df['DAYS_BIRTH'] = (df['DAYS_BIRTH']/(-365)).astype(int)
df['DAYS_EMPLOYED'] = (df['DAYS_EMPLOYED']/(-365)).astype(int)
df.rename(columns = {'DAYS_BIRTH': 'age', 'DAYS_EMPLOYED': 'employment_duration', 'NAME_EDUCATION_TYPE': 'education', 'AMT_INCOME_TOTAL': 'income', 'CNT_CHILDREN': 'children', 'FLAG_OWN_CAR': 'car'}, inplace = True)
df.reset_index(inplace = True, drop = True)
des = df.describe()
des.rename(columns={"CODE_GENDER": "gender", "FLAG_OWN_REALTY": "property", "NAME_INCOME_TYPE": "income_type", "NAME_FAMILY_STATUS": "marital_status", 
                    "NAME_HOUSING_TYPE": "house_type", "FLAG_WORK_PHONE": "work_phone", "FLAG_PHONE": "phone", 
                    "FLAG_EMAIL": "email", "CNT_FAM_MEMBERS": "family_size"}, inplace = True)

des = des.T.drop(columns = ['count'])
