import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
# 내가 알고 싶은 미래
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])
y = np.array([4,5,6,7,8,9,10])

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape) # (7(batch_size), 3(timesteps), 1(feature)) RNN 기본 X 데이터 구조

#2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=10, input_shape=(3,1)))
model.add(Bidirectional(GRU(units=10), input_shape=(3,1))) #Bidirectional => 모델이 아니라 특정 모델에 랩핑 해서 사용 함
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

model.summary()

################# param 개수 ################
'''''''''
RNN : 120
Bidirectional : 240

GRU : 390
Bidirectional :  780

LSTM : 480
Bidirectional :  960
'''''''''
