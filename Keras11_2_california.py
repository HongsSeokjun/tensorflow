import sklearn as sk
#print(sk.__version__) #1.1.3
import tensorflow as tf
#print(tf.__version__) #2.9.3

from tensorflow.python.keras.models import Sequential
import numpy as np
import pandas as pd # 전처리 
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.metrics import r2_score

#1. 데이터
dataset  = fetch_california_housing()
#dataset = pd.DataFrame(data=housing.data, columns=housing.feature_names)
#print(dataset.info())
#exit()
x = dataset.data
y = dataset.target

x = x[0:500]
y = y[0:500]
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size= 0.25,random_state= 6) #6, 21, 36
print(dataset.info())

#print(x.shape)#(20640, 8)
#print(y.shape)#(20640,)
exit()
#2. 모델 구성
model = Sequential()
model.add(Dense(400, input_dim = 8))
model.add(Dense(400))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size= 32)

#4. 평가, 예측
print('#############################')
loss = model.evaluate(x_test,y_test)
result = model.predict(x_test)
print('loss :', loss)
r2 = r2_score(y_test, result)
print('R2 :',r2)

# loss : 0.6286523938179016
# R2 : 0.5459995640086508 #0.59 이상