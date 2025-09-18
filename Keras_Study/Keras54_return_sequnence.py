import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, BatchNormalization

#1. 데이터
# 내가 알고 싶은 미래
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              [8,9,10],
              [9,10,11],
              [10,11,12],
              [20,30,40],
              [30,40,50],
              [30,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70]) # => 80이 나와야함
x = x.reshape(x.shape[0],x.shape[1],1)

print(x.shape) # (7(batch_size), 3(timesteps), 1(feature)) RNN 기본 X 데이터 구조
# x = np.array([[[1],[2],[3]],
#               [[2],[3],[4]],
#               ....
#               [[7],[8],[9]]])

#2. 모델구성
model = Sequential()
model.add(SimpleRNN(30, input_shape=(3,1),return_sequences=True)) # return_sequences=True 차원 유지
model.add(LSTM(10)) # 3차원이지만 아웃풋이 2차원이 된다는 점
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs= 1500, shuffle=False)

#4. 평가 예측
result = model.evaluate(x,y)
print('loss :', result)

x_pred =x_predict.reshape(1,3,1) #(3,) => (1,3,1)
y_pred = model.predict(x_pred)

print('[50,60,70]의 결과 :',y_pred)

#RNN: [50,60,70]의 결과 : [[80.00004]]