#Keras16_validation3_train_test
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
# 1. 데이터
x = np.array(range(1, 170))
y= np.array(range(1, 170))

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.15,random_state=44)
#x_train1, x_val1, y_train1, y_val1 = train_test_split(x_train,y_train, test_size=0.2,shuffle= False)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim = 1, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1))
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 200, batch_size=2,verbose=1 ,validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
result = model.predict([17])
print('loss :', loss)
print('[17]의 예측값 :', result)