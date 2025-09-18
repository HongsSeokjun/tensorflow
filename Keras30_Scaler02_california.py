import sklearn as sk
#print(sk.__version__) #1.1.3
import tensorflow as tf
#print(tf.__version__) #2.9.3

from tensorflow.python.keras.models import Sequential,load_model
import numpy as np
import pandas as pd # 전처리 
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
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
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

path = '.\_save\Keras28_mcp\\02_california\\'
model = load_model(path+'0028-0.4546Keras28_MCP_save_02_california.hdf5')
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

#4. 평가, 예측
print('#############################')
result = model.predict(x_test)
print('result :',result)
r2 = r2_score(y_test, result)
print('R2 :',r2)


# R2 : 0.6300610838696221
#MinMaxScaler
#R2 : -0.44513206641159186
#StandardScaler
#R2 : -0.25099605832638905
#MaxAbsScaler
#R2 : -0.39950880254782595
#RobustScaler
#R2 : -0.21672487381401884
