from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
import pandas as pd

np_path = 'C:\Study25\_data\_save_npy\\'
path = './_data\\Kaggle\\cat_dog\\'
submission_csv = pd.read_csv(path+'sample_submission.csv',index_col=0)

from sklearn.model_selection import train_test_split
#start = time.time()
x_train = np.load(np_path+"Keras50_catdog_x_train100.npy")
y_train = np.load(np_path+"Keras50_catdog_y_train100.npy")
# x_test = np.load(np_path+"Keras50_catdog_x_test100.npy")
# y_test = np.load(np_path+"Keras50_catdog_y_test100.npy")

augment_size = 1000  # 증가시킬 사이즈 
randidx = np.random.randint(x_train.shape[0], size=augment_size)

x_augmented = x_train[randidx].copy() # 4만개의 데이터 copy, copy로 새로운 메모리 할당.
                                      # 서로 영향 x
y_augmented = y_train[randidx].copy()

############# 요기부터 증폭 ###################
datagen = ImageDataGenerator(
    # rescale=1./255, # 0~255 스케일링, 정규화
    horizontal_flip=True, # 수평 뒤집기 <- 데이터 증폭 또는 변환
    # vertical_flip= True, # 수직 뒤집기 <- 데이터 증폭 또는 변환
    width_shift_range=0.1, # 평행이동 10%
    # height_shift_range=0.1, # 수직이동 10%
    # rotation_range=15, # 회전 5도
    # zoom_range=1.2, # 줌 1.2배
    # shear_range=0.7, # 좌표하나를 고정시키고, 다른 몇개의 좌표를 이동시키는 변환(찌부 만들기)
    # fill_mode='nearest',
)

x_augmented = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    save_to_dir='C:\Study25\_data\_save_img\\05_catdog\\',
    save_prefix='catdog',     # 파일 이름 앞부분 설정 가능
    shuffle=False,
).next()[0]