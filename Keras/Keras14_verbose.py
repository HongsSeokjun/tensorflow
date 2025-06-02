# Keras08_train_test1 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])
# print(x.shape) #(10,)
# print(y.shape) #(10,)

#훈련 데이터와 평가 데이터 => 성능 확인
x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

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
          verbose = 2) # 해당 훈련 과정이 다 디버그 찍히는게 비효율적일 수 있다.
#verbose = 0(결과만,침묵), 1(진행바, 디폴트), 2(진행바 없음), 3(에포만 나옴.)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
result = model.predict([11])

print('loss :', loss)
print('[11]의 예측값 :',result)
                       
# loss : 7.579122740649855e-14
# [11]의 예측값 : [[11.]]                      