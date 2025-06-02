import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
# 1. 데이터
x = np.array(range(1, 17))
y= np.array(range(1, 17))

#[실습] 리스트의 슬라이싱으로 10:4:3 으로 나눈다.

#x_1 = x[0:14]
# #print('x_train',x_train)
#x_val = x[14:17]
# #print('x_val',x_val)
# x_test = x[14:]
# #print('x_test',x_test)
#y_1 = y[0:14]
#y_val = y[14:17]
# y_test = y[14:]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.15,random_state=44)
x_train1, x_val1, y_train1, y_val1 = train_test_split(x_train,y_train, test_size=0.2)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim = 1, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1))
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 200, batch_size=2,verbose=1 , validation_data=(x_val1, y_val1))

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
result = model.predict([17])
print('loss :', loss)
print('[17]의 예측값 :', result)