from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization,Conv2D,Flatten,LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score#불균형 데이터일때
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pylab as plt
from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint
#1. 데이터
dataset  = fetch_california_housing()
x = dataset.data
y = dataset.target
#print(x.shape)#(20640, 8)
print(y)
print(np.max(x), np.min(x))

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state= 47)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 8, 1)  # (batch, height, width, channel)
x_test = x_test.reshape(-1, 8, 1)  # (batch, height, width, channel)


#2. 모델구성
model = Sequential()
model.add(LSTM(32,input_shape=(8,1), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
# model.add(Flatten())
model.add(Dense(8,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(
    monitor='val_loss', # 평가 지표로 확인하겠다
    mode= 'min', # 최대값 max, 알아서 찾아줘:auto
    patience=10, # 10번까지 초과해도 넘어가겠다
    restore_best_weights= True # val_loss값이 가장 낮은 값으로 저장 해놓겠다(False시 => 최소값 이후 10번째 값으로 그냥 잡는다.)
)
model.compile(loss='mse',optimizer='adam')
hist = model.fit(x_train, y_train,epochs= 150, batch_size= 4,verbose=2,validation_split=0.1,callbacks=[es])


# 4. 평가 예측

loss = model.evaluate(x_test,y_test)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교

r2 = r2_score(y_test, result)
print('loss :',loss)
print('result :',result)
print('r2 :',r2)
#loss : 0.49591484665870667
#r2 : 0.6179990780467324

# r2 : 0.7363461500847072