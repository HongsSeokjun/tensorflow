# 배치를 적용한거.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6]) #2개 이상의 데이터 List 형태로 하나의 덩어리
# batch 단위로 쪼개서 데이터를 훈련시키면 loss값을 낮출 수 있다.
# ex) 1,2,3,4,5,6을 각각 훈련시키는 것

# 에포는 100으로 고정
# loss 기준 0.32 미만으로 만들것

#2. 모델 구성
model = Sequential()
model.add(Dense(3000,input_dim=1))
model.add(Dense(1))
epochs = 100

#3. 컴파일, 훈련
model.compile(loss ='mse', optimizer='adam')
model.fit(x,y,epochs = epochs, batch_size = 3)
# 너무 작게 잘라도 효율이 떨어질 수 도있다, 인식이 잘 안될 수 있음.

#4. 평가, 예측
loss = model.evaluate(x,y)
print("##########################")
print("epochs :", epochs)
print("loss :", loss)
results = model.predict([6])
print('3의 예측값 :',results)

########################################
# epochs : 100
# loss : 0.32381266355514526
# 1/1 [==============================] - 0s 50ms/step
# 3의 예측값 : [[5.8583455]]
