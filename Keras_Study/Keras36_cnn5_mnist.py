import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

# x reshape -> (60000, 28, 28)
x_train = x_train.reshape(60000, 28, 28,1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)

print(x_train.shape, x_test.shape) #(60000, 28, 28, 1) (10000, 28, 28, 1)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

# print(y_train.shape)
# exit()
#2. 모델 구성
model = Sequential()
model.add(Conv2D(128,(3,3), strides=2 ,input_shape=(28,28,1)))
model.add(Conv2D(filters=64,kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(32,(3,3), activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=50,verbose=1,
                   restore_best_weights= True)

############### mcp 세이브 파일명 만들기 ##############
import datetime
date = datetime.datetime.now()
print(date)
print(type(date))
date = date.strftime('%m%d_%H%M%S')
print(date)
print(type(date)) # <class 'str'>

path = '.\_save\Keras36_cnn5\\'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path,'k36_',date,'_',filename])
#print(filepath)
#exit()
#####################################################
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only= True,
    filepath=filepath
)

start = time.time()
hist = model.fit(x_train,y_train, epochs= 500, batch_size= 512,verbose=2,validation_split=0.1, callbacks=[es,mcp])
end = time.time()
#4. 평가 예측
loss = model.evaluate(x_test,y_test,verbose=1)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교
print('loss :',loss[0])
print('acc :',loss[1])
y_test = y_test.values  #=> 판다스를 넘파이로
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test =  np.argmax(y_test, axis=1)

# acc = accuracy_score(y_test, y_pred)
# print('acc :',round(acc,4))
print('time :',(end-start))

# gpu
# loss : 2.3010332584381104
# acc : 0.11349999904632568
# time : 312.3143141269684
# gpu
# loss : 0.034305837005376816
# acc : 0.9923999905586243
# time : 223.3766438961029

# rtx5090 => 5세대 9성능

# cpu
# loss : 1.3737719058990479
# acc : 0.47130000591278076
# time : 1893.9190304279327