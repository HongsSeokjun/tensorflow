import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D,Dense, Dropout,BatchNormalization,Conv2D,Flatten
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import matplotlib.pylab as plt
import matplotlib
import pandas as pd

train_datagen = ImageDataGenerator(
    rescale=1./255, # 0~255 스케일링, 정규화
)

path_train = 'C:\Study25\_data\\tensor_cert\\rps\\'

# 파일을 열꺼야
xy_train = train_datagen.flow_from_directory(
    path_train,             # 경로
    target_size=(150,150),  # 리사이즈, 사이즈 규격 일치, 큰거는 축소, 작은거는 확대
    batch_size=100,          # 160개의 사진을 (200,200,1) 16번으로 나누는 이유는 메모리 문제!
    class_mode='categorical',   # categorical(다중 원핫o), sparse(다중 원핫x), binary, None(y값 자체가 없이 만들어진다.)
    color_mode='rgb',           # rgb
    shuffle=True,
    seed=333,
) # Found 160 images belonging to 2 classes. //2 classes = ad, normal


import time

end = time.time()
print(xy_train[0][0].shape)
print(xy_train[0][1].shape)
print(len(xy_train)) #250

# print(x_train.shape, y_train.shape) #(100, 200, 200, 3) (100,)
# print('걸린 시간 :',round(end- start,2),'초') #걸린 시간 : 0.39
######### 모든 수치화된 batch데이터를 하나로 합치기 #######
all_x = []
all_y = []
for i in range(len(xy_train)):
    x_batch, y_batch = xy_train[i]
    all_x.append(x_batch)
    all_y.append(y_batch)

###### 리스트를 하나의 numpy 배열로 합친다. ##### (사슬처럼 엮다.)
x = np.concatenate(all_x, axis=0)
y = np.concatenate(all_y, axis=0)
# print('걸린 시간 :',round(end1- end,2),'초') #걸린 시간 : 47.7 초 걸린 시간 : 295.62 초
print("x.shape :", x.shape) #x.shape : (25000, 200, 200, 3)
print("y_shape :", y.shape) #y_shape : (25000,)

np_path = 'C:\Study25\_save\Keras46_rps\\'
np.save(np_path+'Keras46_05_x_train150.npy', arr=x)
np.save(np_path+'Keras46_05_y_train150.npy', arr=y)


