#Keras27_ModelCheckPoint2_load 복붙
import sklearn as sk
print(sk.__version__) #0.24.2 #1.1.3
from sklearn.datasets import load_boston
import numpy as np
from keras.models import Sequential, load_model # 모델 불러오기
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#1. 데이터
dataset = load_boston()
#print(dataset)
#print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

x = dataset.data
y = dataset.target
#print(x.shape, y.shape) #(506, 13) (506,)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state=42)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델 구성 # save weight는 모델이 필요

# model = Sequential([
#     Dense(32, input_dim = 13, activation='relu'),
#     Dense(16, activation='relu'),
#     Dense(8, activation='relu'),
#     Dense(4, activation='relu'),
#     Dense(1)
# ])
path = '.\_save\Keras28_mcp\\01_boston\\'
model = load_model(path+'0058-24.6014_Keras28_boston.hdf5')


# 아직 가중치를 가져오지는 못했다.

model.summary()
#model.save(path+ 'keras26_1_save.h5')

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

loss = model.evaluate(x_test,y_test)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교
r2 = r2_score(y_test, result)
print('loss :',loss)
print('R2 :',r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test,result)
print('Rmse :',rmse)

# loss : 5983904.0
# R2 : -95842.33987709842
# Rmse : 2446.201941423261

#MinMaxScaler
# loss : 10.812653541564941
# R2 : 0.8268152805197155
# Rmse : 3.288259792431024
# StandardScaler
# loss : 473.906494140625
# R2 : -6.590493372349096
# Rmse : 21.769393996296063
# MaxAbsScaler
# loss : 16.292936325073242
# R2 : 0.7390383564698059
# Rmse : 4.036450921250647
#RobustScaler
# loss : 385.67010498046875
# R2 : -5.177223299320543
# Rmse : 19.63848526966292