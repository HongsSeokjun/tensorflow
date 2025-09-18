import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D,Dense, Dropout,BatchNormalization,Conv1D,Flatten
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import matplotlib.pylab as plt
import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split
np_path = 'C:\study25ju\_data\_save_npy\\'

x_train = np.load(np_path+"keras46_rps_02_x_train.npy")
y_train = np.load(np_path+"keras46_rps_02_y_train.npy")
# print(x_train.shape)
# exit()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],x_train.shape[2]*x_train.shape[3])
y_train = pd.get_dummies(y_train)
print(x_train.shape, y_train.shape)  

x_train1, x_test, y_train1, y_test = train_test_split(x_train,y_train,test_size=0.1, random_state= 333,stratify=y_train)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(256, kernel_size=2,input_shape=(200,600), activation='relu'))
model.add(Conv1D(30,2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(16,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(8,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1], activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=20,verbose=1,
                   restore_best_weights= True)

# path = '.\_save\Keras46_rps\\'
# filename = '.hdf5'
# filepath = "".join([path,'k46_6',filename])

# #####################################################
# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_best_only= True,
#     filepath=filepath
# )

hist = model.fit(x_train,y_train, epochs= 1, batch_size= 32,verbose=2,validation_split=0.1, callbacks=[es,])
#4. 평가 예측
loss = model.evaluate(x_test,y_test)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교
print('loss :',loss[0])
print('acc :',loss[1])
y_pred = model.predict(x_test)
y_pred =  (y_pred > 0.5).astype(int)

y_pred = np.argmax(y_pred, axis=1)
y_test_np = y_test.values
images = x_test.reshape(-1, 200, 200,3)

import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i],cmap='gray')
    plt.title(f"예측:{y_pred[i]} / 정답:{np.argmax(y_test_np[i])}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# loss : 0.027814030647277832
# acc : 0.995121955871582
# loss : 1.2745542526245117
# acc : 0.7609755992889404