from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일
model.compile(loss = 'mse', optimizer ='adam')
model.fit(x, y, epochs=9500) #100번째 가중치가 들어가있음

print("#######################################")
#4. 평가 예측
loss = model.evaluate(x,y)
print('로스 : ',loss)
result = model.predict([1,2,3,4,5,6,7])
print("예측값 : ", result)
