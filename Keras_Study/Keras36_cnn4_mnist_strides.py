import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

# x reshape -> (60000, 28, 28)
x_train = x_train.reshape(60000, 28, 28,1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)

print(x_train.shape, x_test.shape) #(60000, 28, 28, 1) (10000, 28, 28, 1)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

# print(y_train.shape)
# exit()
#2. 모델 구성
model = Sequential()
model.add(Conv2D(64,(2,2), strides=2 ,input_shape=(10,10,1))) # strides 보폭
model.add(Conv2D(filters=32,kernel_size=(3,3)))
model.add(Dropout(0.3))
#model.add(Conv2D(16,(3,3)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(16))
model.add(Dropout(0.3))
model.add(Dense(16))
model.add(Dropout(0.3))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.summary()

#  conv2d (Conv2D)             (None, 5, 5, 64)          320

#  conv2d_1 (Conv2D)           (None, 3, 3, 32)          18464 # 2D 사진에서는 (커널높이 * 커널너비 * 입력채널수+1) * 출력으로 계산

#  dropout (Dropout)           (None, 3, 3, 32)          0

#  dropout_1 (Dropout)         (None, 3, 3, 32)          0

#  flatten (Flatten)           (None, 288)               0

#  dense (Dense)               (None, 16)                4624

#  dropout_2 (Dropout)         (None, 16)                0

#  dense_1 (Dense)             (None, 16)                272

#  dropout_3 (Dropout)         (None, 16)                0

#  dense_2 (Dense)             (None, 10)                170

# =================================================================
# Total params: 23,850
# Trainable params: 23,850
# Non-trainable params: 0
# ___________________________________________________