import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D,Dense, Dropout,BatchNormalization,Conv2D,Flatten
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import matplotlib.pylab as plt
import matplotlib
train_datagen = ImageDataGenerator(
    rescale= 1./255
)

test_datagen =  ImageDataGenerator(
    rescale=1./255
)

path_train = 'C:\Study25\_data\image\\brain\\train\\'
path_test ='C:\Study25\_data\image\\brain\\test\\'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(200,200),
    batch_size=160,
    class_mode='binary',    # 
    color_mode='grayscale',
    shuffle=True,
    seed=1,
)

xy_test = test_datagen.flow_from_directory(
    path_test,             # 경로
    target_size=(200,200),  # 리사이즈, 사이즈 규격 일치, 큰거는 축소, 작은거는 확대
    batch_size=120,          # 120개의 사진을 (200,200,1) 12번으로
    class_mode='binary',    # 
    color_mode='grayscale',
    #shuffle=True,
) # Found 120 images belonging to 2 classes.

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

np_path = 'C:\Study25\_data\_save_npy\\'
np.save(np_path+'Keras46_brain_x_train200.npy', arr=x_train)
np.save(np_path+'Keras46_brain_y_train200.npy', arr=y_train)

np.save(np_path+'Keras44_01_x_test200.npy', arr=x_test)
np.save(np_path+'Keras44_01_y_test200.npy', arr=y_test)