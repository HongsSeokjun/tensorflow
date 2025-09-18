from tensorflow.keras.datasets import cifar10
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout,MaxPooling2D,Dense,BatchNormalization,Flatten
import matplotlib.pylab as plt

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)

# 스케일링 2. 정규화 (많이쓴다.) 0 ~ 1
x_train = x_train / 255.0 # 정규화
x_test = x_test / 255.0

print(np.unique(y_train, return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],

y_train = y_train.reshape(50000,)
y_test = y_test.reshape(10000,)
print(pd.value_counts(y_test))
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

model = Sequential()
model.add(Conv2D(128,(3,3),input_shape=(32,32,3), activation='relu'))
model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(filters=64,kernel_size=(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(32,(2,2), activation='relu'))
model.add(MaxPooling2D()) # shape 반으로 절감  (None, 13, 13, 128)   
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(64,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1], activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=60,verbose=1,
                   restore_best_weights= True)

path = '.\_save\Keras39_cifar10\\'
filename = '.hdf5'
filepath = "".join([path,'k39_1',filename])

#####################################################
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only= True,
    filepath=filepath
)
hist = model.fit(x_train,y_train, epochs= 500, batch_size= 512,verbose=2,validation_split=0.1, callbacks=[es,mcp])
#4. 평가 예측
loss = model.evaluate(x_test,y_test,verbose=1)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교
print('loss :',loss[0])
print('acc :',loss[1])
y_test = y_test.values  #=> 판다스를 넘파이로
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test =  np.argmax(y_test, axis=1)

# 이미지 데이터는 원래 shape (32, 32, 3)이므로 다시 꺼내기
images = x_test.reshape(-1, 32, 32,3)

import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i])
    plt.title(f"예측:{y_pred[i]} / 정답:{y_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# loss : 0.5690983533859253
# acc : 0.8190000057220459