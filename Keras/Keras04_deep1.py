from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])
#데이터 정제가 잘 되어있다면 모델구성이나 epoch수를 증가시키면 최적의 가중치를 얻을 수 있다.
#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1 ))#각각은 파라미터/ 하이퍼 파라미터 튜닝
model.add(Dense(50, input_dim=100))
model.add(Dense(400, input_dim=50))
model.add(Dense(50, input_dim=400))
model.add(Dense(1, input_dim=50))

epochs = 300
#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = epochs)

#4. 평가, 예측
loss = model.evaluate(x,y)
print('##########################')
print('epochs :',epochs)
print('loss :',loss)
results = model.predict([6])
print('6의 예측값 :',results)