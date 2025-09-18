import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Conv1D, Flatten,Reshape

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
# 내가 알고 싶은 미래
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])
y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape) #(7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape) # (7(batch_size), 3(timesteps), 1(feature)) RNN 기본 X 데이터 구조
# x = np.array([[[1],[2],[3]],
#               [[2],[3],[4]],
#               ....
#               [[7],[8],[9]]])

#2. 모델구성
model = Sequential()
model.add(Conv1D(filters=10,kernel_size=2,input_shape=(3,1),
                 activation='relu', padding='same')) # 3차원이지만 아웃풋이 3차원이 된다는 점
# model.add(Conv1D(9,2)) # filters, kernel_size(N,2,9)
model.add(Reshape(target_shape=(30)))
model.add(BatchNormalization())
# model.add(Flatten()) #(N,18)
model.add(Dense(10, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))

model.summary()

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs= 800, shuffle=False)

#4. 평가 예측
result = model.evaluate(x,y)
print('loss :', result)

x_pred = np.array([8,9,10]).reshape(1,3,1) #(3,) => (1,3,1)
y_pred = model.predict(x_pred)

print('[8,9,10]의 결과 :',y_pred)
