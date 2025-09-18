from tensorflow.keras.datasets import imdb
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,BatchNormalization,Embedding,Conv1D,Flatten

(x_train,y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000,
)
# print(x_train)
# print(y_train) #[1 0 0 ... 0 1 0]
# print(x_train.shape) #(25000,)
# print(y_train.shape) #(25000,)
# print(np.unique(y_train, return_counts=True)) #[0 1]
# print(pd.value_counts(y_train))
# print('최대길이 :', max(len(i) for i in x_train)) #최대길이 : 2494
# print('최소길이 :', min(len(i) for i in x_train)) #최소길이 : 11
# print('평균길이 :', sum(map(len, x_train))/len(x_train)) #평균길이 : 238.71364
# acc = 0.85 이상
######## 패딩 #######
from tensorflow.keras.preprocessing.sequence import pad_sequences
padding_x = pad_sequences(
                            x_train,
                            padding='pre',
                            maxlen= 240
                         )
padding_test_x = pad_sequences(
                                 x_test,
                                 padding='pre',
                                 maxlen= 240
                              )
print(padding_x.shape) #(25000, 240)

#2. 모델 구성
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim= 500, input_length=240))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv1D(300, kernel_size=2,activation='relu'))
model.add(Conv1D(200, kernel_size=2,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(20,  activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=60,verbose=1,
                   restore_best_weights= True)
hist = model.fit(padding_x,y_train, epochs= 1, batch_size= 32,verbose=2,validation_split=0.1, callbacks=[es,])

#4. 평가 예측
loss = model.evaluate(padding_test_x,y_test,verbose=1)
result = model.predict(padding_test_x) #원래의 y값과 예측된 y값의 비교
# print(result.shape) #(2246, 46)
# result = np.argmax(result, axis=1)
print('loss :',loss[0])
print('acc :',loss[1])
# print('result:',result[:10])

# loss : 0.2980410158634186
# acc : 0.8722000122070312