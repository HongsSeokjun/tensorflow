from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D

#2. 모델구성
model = Sequential()
model.add(Conv2D(10,(2,2), input_shape=(10,10,1),strides=1,padding='same'))

model.add(Conv2D(filters=9, kernel_size=(3,3), strides=1,padding='valid'))#padding='valid' default로 없는것

model.add(Conv2D(8,4))# 4=> (4,4) 알아서 인식

model.summary()