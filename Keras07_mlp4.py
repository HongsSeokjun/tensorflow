from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 데이터
x = np.array([range(10)])  # (1, 10)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [10,9,8,7,6,5,4,3,2,1],
             [9,8,7,6,5,4,3,2,1,0]]) #11,0,-1
#print('x',x)
x = x.T  # (10, 1)
y = y.T  # (10, 3)
#exit()
# 모델
model = Sequential()
model.add(Dense(64, input_dim=1))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(3))  # 출력 3개

# 컴파일 & 학습
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size= 3)  # epoch 늘림

# 평가 및 예측
loss = model.evaluate(x, y)
result = model.predict([[10]])

print("Loss:", loss)
print("Result:")
print(result)