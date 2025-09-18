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
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
np_path = 'C:\Study25\_save\Keras46_gender\\'

x_train = np.load(np_path+"Keras46_07_x_train250.npy")
y_train = np.load(np_path+"Keras46_07_y_train250.npy")


print(x_train.shape, y_train.shape)  #(1027, 100, 100, 3) (1027,)

x_train1, x_test, y_train1, y_test = train_test_split(x_train,y_train,test_size=0.1, random_state= 47)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64,(2,2),input_shape=(250,250,3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(MaxPooling2D()) # shape 반으로 절감  (None, 13, 13, 128)   
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(32,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(16,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=50,verbose=1,
                   restore_best_weights= True)

path = '.\_save\Keras46_gender\\'
filename = '.hdf5'
filepath = "".join([path,'k46_8_1',filename])

#####################################################
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only= True,
    filepath=filepath
)

hist = model.fit(x_train,y_train, epochs= 150, batch_size= 32,verbose=2,validation_split=0.1,
                 callbacks=[es,mcp],class_weight=class_weights,)
#4. 평가 예측
loss = model.evaluate(x_test,y_test)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교
print('loss :',loss[0])
print('acc :',loss[1])
y_pred = model.predict(x_test)
y_pred =  (y_pred > 0.5).astype(int)
images = x_test.reshape(-1, 250, 250,3)

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

# loss : 0.3235923647880554
# acc : 0.921450138092041