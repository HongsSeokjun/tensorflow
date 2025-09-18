from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6]) #2개 이상의 데이터 List 형태로 하나의 덩어리

# 에포는 100으로 고정
# loss 기준 0.32 미만으로 만들것

#2. 모델 구성
model = Sequential()
model.add(Dense(100,input_dim=1,activation='relu'))
model.add(Dense(300))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(1))
epochs = 100

#3. 컴파일, 훈련
model.compile(loss ='mse', optimizer='adam')
model.fit(x,y,epochs = epochs)

#4. 평가, 예측
loss = model.evaluate(x,y)
print("##########################")
print("epochs :", epochs)
print("loss :", loss)
results = model.predict([6])
print('3의 예측값 :',results)
########################################
# epochs : 100
# loss : 0.32385769486427307
# 3의 예측값 : [[3.0307794]]
