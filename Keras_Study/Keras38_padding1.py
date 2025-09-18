# [실습]
# 100,100,3 이미지를
# 10, 10, 11 으로 줄여봐

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D

#2. 모델구성
model = Sequential()
model.add(Conv2D(10,(2,2), input_shape=(100,100,3),strides=1,padding='same'))
model.add(MaxPooling2D()) 
model.add(MaxPooling2D()) 
model.add(Conv2D(filters=10, kernel_size=(2,2), strides=1))
model.add(MaxPooling2D()) 
model.add(Conv2D(11,(3,3)))

model.summary()