#Keras26_5_save_weights 복붙
import sklearn as sk
print(sk.__version__) #0.24.2 #1.1.3
from sklearn.datasets import load_boston
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import tensorflow as tf
#1. 데이터
dataset = load_boston()

x = dataset.data
y = dataset.target
#print(x.shape, y.shape) #(506, 13) (506,)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state=42)

scaler = StandardScaler()
minmaxs = MinMaxScaler()
x_train = minmaxs.fit_transform(x_train)
x_test = minmaxs.transform(x_test)

#2. 모델 구성

# model = Sequential([
#     Dense(32, input_dim = 13, activation='relu'),
#     (Dropout(0.1)),
#     Dense(16, activation='relu'),
#     (Dropout(0.1)),
#     Dense(8, activation='relu'),
#     (Dropout(0.1)),
#     Dense(4, activation='relu'),
#     (Dropout(0.1)),
#     Dense(1)
# ])

# 2-2. 모델구성 (함수형)
input1 = Input(shape=[13,]) # Sequential 모델의 input_shape랑 같음
dense1 = Dense(32, activation='relu')(input1) #ys1 summary에서 이름이 바뀜
drop1 = Dropout(0.1)(dense1)
dense2 = Dense(16,activation='relu')(drop1)
drop2 = Dropout(0.1)(dense2)
dense3 = Dense(8,activation='relu')(drop2)
drop3 = Dropout(0.1)(dense3)
dense4 = Dense(4,activation='relu')(drop3)
drop4 = Dropout(0.1)(dense4)
output1= Dense(1)(drop4)

model = Model(inputs=input1, outputs=output1)

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

start = time.time()
hist = model.fit(x_train,y_train, epochs= 100, batch_size= 32,verbose=2,validation_split=0.1)
end = time.time()

#4. 평가, 훈련
loss = model.evaluate(x_test,y_test)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교
r2 = r2_score(y_test, result)
print('loss :',loss)
print('R2 :',r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test,result)
print('Rmse :',rmse)

# loss : 11.03296184539795
# R2 : 0.8232866200016069
# Rmse : 3.3215902247527516

# loss : 11.241966247558594
# R2 : 0.8199390335562531
# Rmse : 3.3529040489948247