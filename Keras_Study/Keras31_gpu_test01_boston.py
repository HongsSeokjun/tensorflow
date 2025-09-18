#Keras26_5_save_weights 복붙
import sklearn as sk
print(sk.__version__) #0.24.2 #1.1.3
from sklearn.datasets import load_boston
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import tensorflow as tf
#1. 데이터
dataset = load_boston()

x = dataset.data
y = dataset.target
#print(x.shape, y.shape) #(506, 13) (506,)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state=42)

scaler = StandardScaler()
minmaxs = MinMaxScaler()
x_train = minmaxs.fit_transform(x_train)
x_test = minmaxs.transform(x_test)

#2. 모델 구성

model = Sequential([
    Dense(32, input_dim = 13, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1)
])

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

start = time.time()
hist = model.fit(x_train,y_train, epochs= 100, batch_size= 32,verbose=2,validation_split=0.1)
end = time.time()

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    print('GPU 있다')
else:
    print('GPU 없다')


print("걸린시간 :",end-start)

# GPU 없다
# 걸린시간 : 2.6684651374816895
# GPU 있다
# 걸린시간 : 5.644698858261108