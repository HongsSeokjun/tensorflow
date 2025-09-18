import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
import time
import tensorflow as tf
#1. 데이터
dataset = load_diabetes()

x = dataset.data
y = dataset.target

#print(x)
#print(y)
#print(x.shape) #(442, 10)
#print(y.shape) #(442,)
#exit()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=947)#947

#2. 모델 구성
model = Sequential()
model.add(Dense(300, input_dim = 10, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
start = time.time()
hist = model.fit(x_train, y_train,epochs= 100, batch_size= 32,verbose=2,validation_split=0.1)#,class_weight=class_weights,)
end = time.time()
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    print('GPU 있다')
else:
    print('GPU 없다')


print("걸린시간 :",end-start)
# GPU 있다
# 걸린시간 : 5.488444805145264
# GPU 없다
# 걸린시간 : 2.7976737022399902