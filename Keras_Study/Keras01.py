import tensorflow as tf
print(tf.__version__) # 2.9.3
import numpy as np
print(np.__version__) # 1.21.1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])
print(x.shape)#(3,)
#exit()
dense = Dense(1, input_dim=1)
#2. 모델구성
model = Sequential()
model.add(dense)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=5000) # 훈련시키다 => 반복 시키면서 가중치가 쌓인다.

#4. 평가, 예측
result = model.predict([4])
print('4의 예측값 : ',result)
#print(dense.get_weights())
