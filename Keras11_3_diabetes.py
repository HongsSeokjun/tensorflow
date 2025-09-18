import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
dataset = load_diabetes()

x = dataset.data
y = dataset.target

#print(x)
#print(y)
#print(x.shape) #(442, 10)
#print(y.shape) #(442,)
#exit()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=947)#947

#2. 모델 구성
model = Sequential()
model.add(Dense(300, input_dim = 10))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer ='adam')
model.fit(x_train,y_train, epochs=100 , batch_size =32)
#4. 평가, 예측
print('#############################')
loss = model.evaluate(x_test,y_test)
result = model.predict(x_test)
print('loss :', loss)
def rmse(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print('Rmse :',rmse(y_test,result))
print('R2 :',r2_score(y_test, result))

#0.62 이상
# loss : 2438.707763671875
# Rmse : 49.383274959112946
#R2 : 0.6213395119933437