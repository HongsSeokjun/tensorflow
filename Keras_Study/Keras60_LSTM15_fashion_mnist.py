# CNN -> DNN

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,BatchNormalization,LSTM
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import matplotlib.pylab as plt

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

# 스케일링
x_train = x_train/255.
x_test = x_test/255. 
print(np.max(x_train), np.min(x_train)) #1.0 0.0
print(np.max(x_test), np.min(x_test)) #1.0 0.0

# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]) #(60000, 784)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])  #(10000, 784)

y_train = pd.get_dummies(y_train) #(60000, 10)
y_test = pd.get_dummies(y_test) #(10000, 10)

#print(x_train.shape)
# print(y_train.shape)
# exit()
#2. 모델 구성
model = Sequential()
model.add(LSTM(256, input_shape=(28,28), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128,   activation='relu'))#activation=LeakyReLU(alpha=0.1)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(56,   activation='relu'))#activation=LeakyReLU(alpha=0.1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(28,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(10,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(y_train.shape[1], activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=60,verbose=1,
                   restore_best_weights= True)
path = '.\_save\Keras40_fashion\\'
filename = '.hdf5'
filepath = "".join([path,'k40_',filename])

#####################################################
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only= True,
    filepath=filepath
)
import time
start = time.time()
hist = model.fit(x_train,y_train, epochs= 1, batch_size= 512,verbose=2,validation_split=0.1, callbacks=[es,mcp])
end = time.time()
#4. 평가 예측
loss = model.evaluate(x_test,y_test,verbose=1)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교
print('loss :',loss[0])
print('acc :',loss[1])

print('시간 :',round(end - start,3))

y_pred = np.argmax(result, axis=1)
y_test =  np.argmax(y_test.values, axis=1)

# 이미지 데이터는 원래 shape (32, 32, 3)이므로 다시 꺼내기
images = x_test.reshape(-1, 28, 28,1)

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



#gpu cpu 시간도 체크


# gpu
# loss : 0.3351460099220276
# acc : 0.8928999900817871
# 시간 : 381.507

# loss : 1.6262463331222534
# acc : 0.4961000084877014
# 시간 : 21.002