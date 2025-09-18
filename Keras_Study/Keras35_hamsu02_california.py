import sklearn as sk
#print(sk.__version__) #1.1.3
import tensorflow as tf
#print(tf.__version__) #2.9.3

from tensorflow.python.keras.models import Sequential,Model
import numpy as np
import pandas as pd # 전처리 
from tensorflow.python.keras.layers import Dense, Dropout,Input
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.metrics import r2_score

#1. 데이터
dataset  = fetch_california_housing()
#dataset = pd.DataFrame(data=housing.data, columns=housing.feature_names)
#print(dataset.info())
#exit()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size= 0.2,random_state= 36) #6, 21, 36
#print(dataset.info())

#print(x.shape)#(20640, 8)
#print(y.shape)#(20640,)
#exit()
#2. 모델 구성
# model = Sequential()
# model.add(Dense(400, input_dim = 8, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(400, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(300, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(300, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(300, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(20, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1))

# 2-2. 모델구성 (함수형)
input1 = Input(shape=[8,]) # Sequential 모델의 input_shape랑 같음
dense1 = Dense(400, activation='relu')(input1) #ys1 summary에서 이름이 바뀜
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(400,activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(300,activation='relu')(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(300,activation='relu')(drop3)
drop4 = Dropout(0.3)(dense4)
dense5 = Dense(300,activation='relu')(drop4)
drop5= Dropout(0.3)(dense5)
dense6 = Dense(50,activation='relu')(drop5)
drop6 = Dropout(0.3)(dense6)
dense7 = Dense(20,activation='relu')(drop6)
drop7 = Dropout(0.3)(dense7)
output1= Dense(1)(drop7)

model = Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(
    monitor='val_loss', # 평가 지표로 확인하겠다
    mode= 'min', # 최대값 max, 알아서 찾아줘:auto
    patience=10, # 10번까지 초과해도 넘어가겠다
    restore_best_weights= True # val_loss값이 가장 낮은 값으로 저장 해놓겠다(False시 => 최소값 이후 10번째 값으로 그냥 잡는다.)
)
import datetime
date = datetime.datetime.now()
print(date)
print(type(date))
date = date.strftime('%m%d_%H%M%S')
print(date)
print(type(date)) # <class 'str'>

# filename = '{epoch:04d}-{val_loss:.4f}Keras28_MCP_save_02_california.hdf5'

# path = '.\_save\Keras28_mcp\\02_california\\' #'C:\Study25\_save\Keras28_mcp\\01_boston\\'
# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     save_best_only= True,
#     filepath=path+filename
# )

hist = model.fit(x_train, y_train, epochs=100, batch_size= 16,validation_split=0.1,callbacks=[es])

#4. 평가, 예측
print('#############################')
result = model.predict(x_test)
r2 = r2_score(y_test, result)
print('R2 :',r2)
print(hist.history['loss'])
print('val_loss',hist.history['val_loss'])
# import matplotlib.pylab as plt
# import matplotlib
#R2 : -0.0001252602104491185