from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
import pandas as pd
(x_train, y_train), (x_test, y_test) =  cifar10.load_data()

x_train = x_train/255.  #(50000, 32, 32, 3) (50000, 1)
x_test = x_test/255. #(10000, 32, 32, 3) (10000, 1)

augment_size = 10000  # 증가시킬 사이즈 
############# 요기부터 증폭 ###################
datagen = ImageDataGenerator(
    rescale=1./255, # 0~255 스케일링, 정규화
    horizontal_flip=True, # 수평 뒤집기 <- 데이터 증폭 또는 변환
    # vertical_flip= True, # 수직 뒤집기 <- 데이터 증폭 또는 변환
    width_shift_range=0.1, # 평행이동 10%
    height_shift_range=0.1, # 수직이동 10%
    rotation_range=15, # 회전 5도
    # zoom_range=1.2, # 줌 1.2배
    # shear_range=0.7, # 좌표하나를 고정시키고, 다른 몇개의 좌표를 이동시키는 변환(찌부 만들기)
    # fill_mode='nearest',
)
y_train = y_train.reshape(-1,)

randidx = np.random.randint(x_train.shape[0], size=augment_size)
# => 같은 내용 np.random.randint(60000, 40000) 0~ 60000의 숫자가 4만개

print(randidx) #[14422 32911 45175 ... 18721 38642 34262]
print(np.min(randidx), np.max(randidx)) # 0 59997

x_augmented = x_train[randidx].copy() # 4만개의 데이터 copy, copy로 새로운 메모리 할당.
                                      # 서로 영향 x
y_augmented = y_train[randidx].copy()
print(x_augmented)
print(x_augmented.shape) #(40000, 28, 28)

# x_augmented = x_augmented.reshape(
#     x_augmented.shape[0],
#     x_augmented.shape[1],
#     x_augmented.shape[2],
#     3,
# )


x_augmented = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

# print(x_augmented.shape) #(40000, 28, 28, 1)

# x_train = x_train.reshape(-1,32,32,3)
# x_test = x_test.reshape(-1,32,32,3)

# print(x_train.shape)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)
print(pd.value_counts(y_test))
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


path = '.\_save\Keras39_cifar10\\'
model = load_model(path+'k39_1.hdf5')
#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=50,verbose=1,
                   restore_best_weights= True)
hist = model.fit(x_train,y_train, epochs= 500, batch_size= 512,verbose=2,validation_split=0.1, callbacks=[es])
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])

loss = model.evaluate(x_test,y_test,verbose=1)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교
print('loss :',loss[0])
print('acc :',loss[1])
y_test = y_test.values  #=> 판다스를 넘파이로
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test =  np.argmax(y_test, axis=1)



# gpu
# loss : 0.5690983533859253
# acc : 0.8190000057220459

# loss : 0.6623592972755432
# acc : 0.7885000109672546