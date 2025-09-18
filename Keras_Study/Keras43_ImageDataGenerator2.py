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
    rescale=1./255, # 0~255 스케일링, 정규화
    # horizontal_flip=True, # 수평 뒤집기 <- 데이터 증폭 또는 변환
    # vertical_flip= True, # 수직 뒤집기 <- 데이터 증폭 또는 변환
    # width_shift_range=0.1, # 평행이동 10%
    # height_shift_range=0.1, # 수직이동 10%
    # rotation_range=5, # 회전 5도
    # zoom_range=1.2, # 줌 1.2배
    # shear_range=0.7, # 좌표하나를 고정시키고, 다른 몇개의 좌표를 이동시키는 변환(찌부 만들기)
    #fill_mode='nearest',
)

test_datagen = ImageDataGenerator(
    rescale=1./255, # 0~255 스케일링, 정규화
) # 평가 데이터는 변환하지 않는다.

path_train = 'C:\Study25\_data\image\\brain\\train\\'
path_test ='C:\Study25\_data\image\\brain\\test\\'

# 파일을 열꺼야
xy_train = train_datagen.flow_from_directory(
    path_train,             # 경로
    target_size=(200,200),  # 리사이즈, 사이즈 규격 일치, 큰거는 축소, 작은거는 확대
    batch_size=160,          # 160개의 사진을 (200,200,1) 16번으로 나누는 이유는 메모리 문제!
    class_mode='binary',    # 
    color_mode='grayscale',
    shuffle=True,
    seed=1,
) # Found 160 images belonging to 2 classes. //2 classes = ad, normal

xy_test = test_datagen.flow_from_directory(
    path_test,             # 경로
    target_size=(200,200),  # 리사이즈, 사이즈 규격 일치, 큰거는 축소, 작은거는 확대
    batch_size=120,          # 120개의 사진을 (200,200,1) 12번으로
    class_mode='binary',    # 
    color_mode='grayscale',
    #shuffle=True,
) # Found 120 images belonging to 2 classes.

# print(xy_train[0][0])  #x 데이터(10, 200, 200, 1)
#print(xy_train[0][1])  #y 데이터(10,)
x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]
# plt.figure(figsize=(12, 4))
# plt.imshow(x_train[0])
# plt.show()
# exit()
# print(x_train.shape) #(160, 200, 200, 1)
# print(y_train.shape) #(160,)
# print(x_test.shape) #(120, 200, 200, 1)
# print(y_test.shape) #(120,)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64,(2,2),input_shape=(200,200,1), activation='relu'))
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

path = '.\_save\Keras43_Image\\'
filename = '.hdf5'
filepath = "".join([path,'k43_',filename])

#####################################################
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only= True,
    filepath=filepath
)

hist = model.fit(x_train,y_train, epochs= 250, batch_size= 4,verbose=2,validation_split=0.1, callbacks=[es,mcp])
#4. 평가 예측
loss = model.evaluate(x_test,y_test)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교
print('loss :',loss[0])
print('acc :',loss[1])
y_pred = model.predict(x_test)
y_pred =  (y_pred > 0.5).astype(int)
images = x_test.reshape(-1, 200, 200)


import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i],cmap='gray')
    plt.title(f"예측:{y_pred[i]} / 정답:{y_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# loss : 0.07195821404457092
# acc : 1.0