import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5],
              [6,7,8,9,10]])#.T
y = np.array([1,2,3,4,5])
x = np.transpose(x)
#x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]]) 입력으로 넣으려고 했던 값

print(x.shape) #(5, 2)
print(y.shape) #(5,)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=2)) #행 무시, 열 우선@@
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
result = model.predict([[6,11]]) # (1,2) 입력하는 적합한 데이터가 많을 수록 loss값이 줄어든다.
print('loss :', loss)
print('[6,11]의 예측값 :',result)