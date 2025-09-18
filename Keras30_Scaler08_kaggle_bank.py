# https://www.kaggle.com/competitions/playground-series-s4e1/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Dropout,BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.utils import class_weight
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib
# 1. 데이터
path = './_data/Kaggle/bank/'

train_csv = pd.read_csv(path + 'train.csv',index_col=0)
test_csv = pd.read_csv(path + 'test.csv',index_col=0)
submission_csv = pd.read_csv(path+'sample_submission.csv',index_col=0)

#print(train_csv)#(165034, 13)
#print(train_csv.head(10)) # default 5개 => 원하는 값 위에서 개수까지 확인 가능
#print(train_csv.tail()) # default 5개 => 원하는 값 아래서 개수까지 확인 가능
#print(train_csv.isna().sum()) # 0
le_geo = LabelEncoder() # 클래스를 인스턴스 한다.
le_gender = LabelEncoder()
#print(train_csv.columns)
# Index(['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
#        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
#        'EstimatedSalary', 'Exited'],

train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le_gender.fit_transform(train_csv['Gender'])
print(train_csv['Geography'])
print(train_csv['Geography'].value_counts()) #pandas
# 0    94215
# 2    36213
# 1    34606
print(train_csv['Gender'].value_counts()) #pandas np.unique(data, return_counts=True)
# 1    93150
# 0    71884
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gender.transform(test_csv['Gender'])

train_csv = train_csv.drop(['CustomerId','Surname'],axis= 1)
test_csv = test_csv.drop(['CustomerId','Surname'], axis= 1)
print(train_csv.columns)
print(test_csv)
#exit()

#corr = train_csv.corr()  # 변수들 간 상관관계 계산
#plt.figure(figsize=(10,8))
#sns.boxplot(x=train_csv['Age'])
#sns.heatmap(corr, annot=True, cmap='coolwarm')  # annot=True는 숫자 표시
#plt.show()

#exit()
train_csv['Balance'] = train_csv['Balance'].replace(0, train_csv['Balance'].mean())
test_csv['Balance'] = test_csv['Balance'].replace(0, test_csv['Balance'].mean())
#train_csv.dropna()

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.1, random_state= 42)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# x_train[['CreditScore','Age','Tenure','Balance','EstimatedSalary']] = scaler.fit_transform(x_train[['CreditScore','Age','Tenure','Balance','EstimatedSalary']])        # train 데이터에 맞춰서 스케일링
# x_test[['CreditScore','Age','Tenure','Balance','EstimatedSalary']]= scaler.transform(x_test[['CreditScore','Age','Tenure','Balance','EstimatedSalary']]) # test 데이터는 transform만!
# test_csv[['CreditScore','Age','Tenure','Balance','EstimatedSalary']] = scaler.transform(test_csv[['CreditScore','Age','Tenure','Balance','EstimatedSalary']])

path = '.\_save\Keras28_mcp\\08_kaggle_bank\\'
model = load_model(path+'0029-0.1203Keras28_MCP_save_08_kaggle_bank.hdf5')
#3. 컴파일, 훈련
results = model.evaluate(x_test, y_test)
print(results) 
#[0.034758370369672775, 0.9824561476707458]

#print('loss = ',results[0])
#print('acc = ', round(results[1],4)) # 반올림
y_predict = model.predict(x_test)
y_predict =  (y_predict > 0.5).astype(int)
from sklearn.metrics import accuracy_score # 이진만 받을 수 있다
accuracy_score = accuracy_score(y_test, y_predict)
#[0.7896267771720886, 0.21037323772907257]
#MinMaxScaler
#[0.1458790898323059, 0.8313742280006409]
#StandardScaler
#[0.12178796529769897, 0.835433840751648]
#MaxAbsScaler
#[0.15820065140724182, 0.8191347718238831]
#RobustScaler
#[0.13282983005046844, 0.8390693068504333]