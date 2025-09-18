import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, BatchNormalization

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
# 시계열 데이터는 y 값을 주지 않기 때문에 x와 y를 직접 나눠 줘야한다.
# 타임스탭을 내가 3으로 지정해서 만든 것
# 내가 알고 싶은 미래를 찾는것
# [1,2,3] => 4 , [2,3,4] => 5, [3,4,5] => 6
# ... [8,9,10] => 11 이란 데이터가 없기에

print(x.shape, y.shape) #(7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape) # (7(batch_size), 3(timesteps), 1(feature)) RNN 기본 X 데이터 구조
# x = np.array([[[1],[2],[3]],
#               [[2],[3],[4]],
#               ....
#               [[7],[8],[9]]])

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(18, input_shape=(3,1),activation='relu')) # 3차원이지만 아웃풋이 2차원이 된다는 점
# model.add(SimpleRNN(units=18, input_shape=(3,1),activation='relu')) # 3차원이지만 아웃풋이 2차원이 된다는 점
model.add(LSTM(units=18, input_shape=(3,1),activation='relu')) # 3차원이지만 아웃풋이 2차원이 된다는 점
# model.add(GRU(units=18, input_shape=(3,1),activation='relu')) # 3차원이지만 아웃풋이 2차원이 된다는 점
model.add(BatchNormalization())
model.add(Dense(10, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs= 800, shuffle=False)

#4. 평가 예측
result = model.evaluate(x,y)
print('loss :', result)

x_pred = np.array([8,9,10]).reshape(1,3,1) #(3,) => (1,3,1)
y_pred = model.predict(x_pred)

print('[8,9,10]의 결과 :',y_pred)

#RNN: [8,9,10]의 결과 : [11.0000] //[8,9,10]의 결과 : [[10.918682]]
#LSTM [8,9,10]의 결과 : [[10.918979]]
#GRU [8,9,10]의 결과 : [[11.019848]]