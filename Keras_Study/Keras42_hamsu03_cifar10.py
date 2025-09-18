import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D, Dense, Conv2D, Flatten, Dropout,BatchNormalization,Input
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)

# 스케일링 2. 정규화 (많이쓴다.) 0 ~ 1
x_train = x_train / 255.0 # 정규화
x_test = x_test / 255.0

y_train = y_train.reshape(50000,)
y_test = y_test.reshape(10000,)
print(pd.value_counts(y_test))
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


input1 = Input(shape=[32,32,3]) # Sequential 모델의 input_shape랑 같음
Conv2D1 = Conv2D(128,(3,3), activation='relu')(input1) #ys1 summary에서 이름이 바뀜
Max1 = MaxPooling2D()(Conv2D1) # shape 반으로 절감  (None, 13, 13, 128)
Batch = BatchNormalization()(Max1)
drop = Dropout(0.3)(Batch)
Conv2D2 = Conv2D(filters=64,kernel_size=(3,3), activation='relu')(drop)
Batch1 = BatchNormalization()(Conv2D2)
drop1 = Dropout(0.3)(Batch1)
Conv2D3 = Conv2D(32,(2,2), activation='relu')(drop1)
Max2 = MaxPooling2D()(Conv2D3)
Batch2 = BatchNormalization()(Max2)
drop2 = Dropout(0.3)(Batch2)
flott1 = Flatten()(drop2)
dense3 = Dense(64, activation='relu')(flott1) #ys1 summary에서 이름이 바뀜
Batch3 = BatchNormalization()(dense3)
drop3 = Dropout(0.2)(Batch3)
dense4 = Dense(32,activation='relu')(drop3)
Batch4 = BatchNormalization()(dense4)
drop4 = Dropout(0.2)(Batch4)
output1= Dense(y_train.shape[1], activation='softmax')(drop4)

model = Model(inputs=input1, outputs=output1)



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


