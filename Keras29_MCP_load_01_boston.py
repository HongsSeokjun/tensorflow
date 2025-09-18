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
from sklearn.preprocessing import StandardScaler,MinMaxScaler

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

scaler = StandardScaler()
minmaxs = MinMaxScaler()
x_train = minmaxs.fit_transform(x_train)
x_test = minmaxs.transform(x_test)

# print(np.min(x_train), np.max(x_train))
# print(np.min(x_test), np.max(x_test))

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

# loss : 7.897532939910889
# R2 : 0.8735063348062659
# Rmse : 2.8102549739866145