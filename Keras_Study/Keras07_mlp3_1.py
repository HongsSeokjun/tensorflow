import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10),range(21,31), range(201,211)]) #10,21,201
print(x.shape) #(10, 3)
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1]]) # x의 데이터값 수랑 y 수를 맞춰야한다.
x = x.T #(10, 3)
print(x.shape)
y = y.T #(10, 2)
print(y.shape)

# [실습]
# loss와[[10,31,211]]을 예측하시오.

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim =3)) #0,3의 가중치
model.add(Dense(500))
model.add(Dense(400))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(2))
#3. 컴파일, 훈련
model.compile(loss ='mse', optimizer ='adam')
model.fit(x,y, epochs =100, batch_size=2)

#4. 예측, 평가
loss = model.evaluate(x,y)
result = model.predict([[10,31,211]])
print('loss :', loss)
print('[10,2,-1]의 예측값 :',result)
#loss : 0.007386660668998957
# [10,2,-1]의 예측값 : [[10.827952   -0.10587219]]
