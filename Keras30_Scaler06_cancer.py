#https://dacon.io/competitions/official/236488/overview/description

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization,LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,RobustScaler,OneHotEncoder,StandardScaler
from sklearn.utils import class_weight
import matplotlib.pylab as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
#from xgboost import XGBClassifier
# 1. 데이터
path = './_data/dacon/Cancer/'

train_csv = pd.read_csv(path+'train.csv',index_col=0)
test_csv =pd.read_csv(path+'test.csv', index_col=0)
submission_csv = pd.read_csv(path+'sample_submission.csv',index_col=0)

#print(train_csv.shape) #(87159, 15)
#print(test_csv.shape) #(46204, 14)
# print(train_csv.columns)
# Index(['Age', 'Gender', 'Country', 'Race', 'Family_Background',
#        'Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Weight_Risk',
#        'Diabetes', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result',
#        'Cancer'],

label_cols = ['Gender', 'Country','Race','Family_Background','Smoke',
              'Weight_Risk','Diabetes','Iodine_Deficiency','Radiation_History']
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col] = le.transform(test_csv[col])
    label_encoders[col] = le  # 나중에 inverse_transform 할 때 쓰기 위해 저장
    
# train_csv['Race'] = label_encoders['Race'].inverse_transform(train_csv['Race'])
# print(train_csv['Race'])
# exit()
#from sklearn.preprocessing import OneHotEncoder
# train_csv = pd.get_dummies(train_csv, columns=['Race'])
# test_csv = pd.get_dummies(test_csv, columns=['Race'])

x = train_csv.drop(['Cancer','Diabetes'], axis=1)#'Diabetes'
#x = train_csv[['Gender','Country','Smoke','Weight_Risk','Diabetes','Iodine_Deficiency','Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']]
y = train_csv['Cancer']
#test_csv = test_csv[['Gender','Country','Smoke','Weight_Risk','Diabetes','Iodine_Deficiency','Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']]
test_csv = test_csv.drop(['Diabetes'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.01, random_state= 190,stratify=y)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# x_train[['Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']] = scaler.fit_transform(x_train[['Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']])        # train 데이터에 맞춰서 스케일링
# x_test[['Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']]= scaler.transform(x_test[['Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']]) # test 데이터는 transform만!
# test_csv[['Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']] = scaler.transform(test_csv[['Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']])



path = '.\_save\Keras28_mcp\\06_cancer\\'
model = load_model(path+'0073-0.5248Keras28_MCP_save_06_cancer.hdf5')
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

#4. 평가, 예측
print('#############################')
results = model.evaluate(x_test, y_test)
print(results)

#_predict = model.predict(x_test)
#y_predict =  (y_predict > 0.5).astype(int)
y_predict = model.predict(x_test)
y_predict =  (y_predict > 0.5).astype(int)
f1_score1 = f1_score(y_test, y_predict)
print('f1_score :', f1_score1)
#f1_score : 0.0
#MinMaxScaler
#f1_score : 0.5544554455445544
#StandardScaler
#f1_score : 0.5500000000000002
#MaxAbsScaler
#f1_score : 0.5544554455445544
#RobustScaler
#f1_score : 0.5544554455445544