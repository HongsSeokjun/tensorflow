#validation 검증

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터 
# traindata 3등분 => train(공부)/ validation(문제집) // test(평가)
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])
# print(x.shape) #(10,)
# print(y.shape) #(10,)

#훈련 데이터와 평가 데이터 => 성능 확인
x_train = np.array([1,2,3,4,5,6,])
y_train = np.array([1,2,3,4,5,6,])

x_val = np.array([7,8])
y_val = np.array([7,8])

x_test = np.array([9,10])
y_test = np.array([9,10])

#exit()

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim = 1))
model.add(Dense(400))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss ='mse', optimizer ='adam')
model.fit(x_train,y_train,epochs =300, batch_size = 4,
          validation_data=(x_val, y_val)
          ) # validation_data 과적합을 방지 (가중치 갱신에는 영향을 주지 않는다.)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
result = model.predict([11])

print('loss :', loss)
print('[11]의 예측값 :',result)
                       
# loss : 7.579122740649855e-14 => 최종 지표중 하나, val_loss => 훈련이 제대로 안되고 있다는 것
# [11]의 예측값 : [[11.]]                      
# validation_data => 평가 지표 보다 중요한 부분(loss값이 줄지 않아도, validation_data)