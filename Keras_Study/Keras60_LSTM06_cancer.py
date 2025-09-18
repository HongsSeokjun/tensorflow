from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization,LSTM,Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score#불균형 데이터일때
from sklearn.preprocessing import StandardScaler,LabelEncoder
import tensorflow as tf
import matplotlib.pylab as plt
from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint
#1. 데이터
path = './_data/dacon/Cancer/'

train_csv = pd.read_csv(path+'train.csv',index_col=0)
test_csv =pd.read_csv(path+'test.csv', index_col=0)
submission_csv = pd.read_csv(path+'sample_submission.csv',index_col=0)

label_cols = ['Gender', 'Country','Race','Family_Background','Smoke',
              'Weight_Risk','Diabetes','Iodine_Deficiency','Radiation_History']
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col] = le.transform(test_csv[col])
    label_encoders[col] = le  # 나중에 inverse_transform 할 때 쓰기 위해 저장
    

x = train_csv.drop(['Cancer','Diabetes'], axis=1)#'Diabetes'
#x = train_csv[['Gender','Country','Smoke','Weight_Risk','Diabetes','Iodine_Deficiency','Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']]
y = train_csv['Cancer']
#test_csv = test_csv[['Gender','Country','Smoke','Weight_Risk','Diabetes','Iodine_Deficiency','Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']]
test_csv = test_csv.drop(['Diabetes'], axis=1)
print(x.shape)#(10886, 8)

x = x.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state= 47)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 13, 1)  # (batch, height, width, channel)
x_test = x_test.reshape(-1, 13, 1)  # (batch, height, width, channel)


#2. 모델구성
model = Sequential()
model.add(LSTM(32,input_shape=(13,1), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
# model.add(Flatten())
model.add(Dense(8,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(
    monitor='val_loss', # 평가 지표로 확인하겠다
    mode= 'min', # 최대값 max, 알아서 찾아줘:auto
    patience=10, # 10번까지 초과해도 넘어가겠다
    restore_best_weights= True # val_loss값이 가장 낮은 값으로 저장 해놓겠다(False시 => 최소값 이후 10번째 값으로 그냥 잡는다.)
)
model.compile(loss='binary_crossentropy',optimizer='adam')
hist = model.fit(x_train, y_train,epochs= 1, batch_size= 4,verbose=2,validation_split=0.1,callbacks=[es])


# 4. 평가 예측

loss = model.evaluate(x_test,y_test)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교

r2 = r2_score(y_test, result)
print('loss :',loss)
print('result :',result)
print('r2 :',r2)
#r2 : 0.051396105399651004
# r2 : 0.0018336426494006686