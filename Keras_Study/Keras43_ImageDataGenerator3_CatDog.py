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
    fill_mode='nearest',
)

test_datagen = ImageDataGenerator(
    rescale=1./255, # 0~255 스케일링, 정규화
) # 평가 데이터는 변환하지 않는다.
path = './_data\\Kaggle\\cat_dog\\'
path_train = 'C:\\Study25\\_data\\Kaggle\\cat_dog\\train2\\'
path_test ='C:\\Study25\\_data\\Kaggle\\cat_dog\\test2\\'
submission_csv = pd.read_csv(path+'sample_submission.csv',index_col=0)

# 파일을 열꺼야
xy_train = train_datagen.flow_from_directory(
    path_train,             # 경로
    target_size=(100,100),  # 리사이즈, 사이즈 규격 일치, 큰거는 축소, 작은거는 확대
    batch_size=32,          # 160개의 사진을 (200,200,1) 16번으로 나누는 이유는 메모리 문제!
    class_mode='binary',    # 
    color_mode='rgb',
    shuffle=True,
    seed=1,
) # Found 160 images belonging to 2 classes. //2 classes = ad, normal

xy_train1 = train_datagen.flow_from_directory(
    path_train,             # 경로
    target_size=(100,100),  # 리사이즈, 사이즈 규격 일치, 큰거는 축소, 작은거는 확대
    batch_size=32,          # 160개의 사진을 (200,200,1) 16번으로 나누는 이유는 메모리 문제!
    class_mode='binary',    # 
    color_mode='rgb',
    shuffle=True,
    seed=1,
)# Found 160 images belonging to 2 classes. //2 classes = ad, normal


xy_submit = test_datagen.flow_from_directory(
    path_test,             # 경로
    target_size=(100,100),  # 리사이즈, 사이즈 규격 일치, 큰거는 축소, 작은거는 확대
    batch_size=32,          # 120개의 사진을 (200,200,1) 12번으로
    class_mode='binary',    # 
    color_mode='rgb',
) # Found 120 images belonging to 2 classes.


# print(xy_train[0][0])  #x 데이터(10, 200, 200, 1)
# print(xy_train[0][1])  #y 데이터(10,)
# x_train = xy_train[0][0]
# y_train = xy_train[0][1]
# x_test = xy_submit[0][0]
# y_test =xy_submit[0][1]
#y_submit = xy_submit[0][0]
# print(y_submit)
# exit()
# print(x_train[0][0].shape) #(160, 200, 200, 1) Found 25000 images belonging to 2 classes.
# print(y_train.shape) #(160,)
# print(x_test.shape) #(120, 200, 200, 1)
# print(y_test.shape) #(120,)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64,(2,2),input_shape=(100,100,3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(MaxPooling2D()) # shape 반으로 절감  (None, 13, 13, 128)   
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(32,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(8,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=60,verbose=1,
                   restore_best_weights= True)

# path = '.\_save\Keras43_Image\\'
# filename = '.hdf5'
# filepath = "".join([path,'k43_',filename])

# #####################################################
# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_best_only= True,
#     filepath=filepath
# )

hist = model.fit(
    xy_train,
    validation_data=xy_train1,
    validation_steps=xy_train1.samples // xy_train1.batch_size,  # ✔️ 정수
    steps_per_epoch=xy_train.samples // xy_train.batch_size,     # ✔️ train에도 이걸 넣는 게 좋아요
    epochs=100,
    verbose=2,
    callbacks=[es]
)
#4. 평가 예측
#loss = model.evaluate(x_test,y_test)
import math
result = model.predict(xy_submit, steps=math.ceil(xy_submit.samples / xy_submit.batch_size))
# print('loss :',loss[0])
# print('acc :',loss[1])

y_submit = result#model.predict(x_test)
print(y_submit)
#y_submit =  (y_submit > 0.5).astype(int)
submission_csv['label'] = y_submit

#################### csv파일 만들기 #########################
submission_csv.to_csv(path + 'submission_0613_15.csv') # CSV 만들기.

# images = x_test.reshape(-1, 500, 500)

# import matplotlib
# matplotlib.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize=(12, 4))
# for i in range(10):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(images[i],cmap='gray')
#     plt.title(f"예측:{y_submit[i]} / 정답:{y_test[i]}")
#     plt.axis('off')
# plt.tight_layout()
# plt.show()

# loss : 0.07195821404457092
# acc : 1.0