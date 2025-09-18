


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,1.9],
              [9,8,7,6,5,4,3,2,1,0]])#.T
y = np.array([1,2,3,4,5,6,7,8,9,10])
x = np.transpose(x)

print(x.shape) #(10, 3)
print(y.shape) #(10,)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=3)) #행 무시, 열 우선@@
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

epochs = 100
#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y, epochs = epochs, batch_size=1)

# #4. 평가, 예측
loss = model.evaluate(x,y)
result = model.predict([[11,2.0,-1]]) # (1,2) 입력하는 적합한 데이터가 많을 수록 loss값이 줄어든다.
print('loss :', loss)
print('[10,2,-1]의 예측값 :',result)