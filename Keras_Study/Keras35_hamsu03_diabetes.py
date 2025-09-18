import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense,Dropout,Input
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
dataset = load_diabetes()

x = dataset.data
y = dataset.target

#print(x)
#print(y)
#print(x.shape) #(442, 10)
#print(y.shape) #(442,)
#exit()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=947)#947

#2. 모델 구성
# model = Sequential()
# model.add(Dense(300, input_dim = 10, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(200, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(20, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1))

input1 = Input(shape=[10,]) # Sequential 모델의 input_shape랑 같음
dense1 = Dense(300, activation='relu')(input1) #ys1 summary에서 이름이 바뀜
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(200,activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(100,activation='relu')(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(50,activation='relu')(drop3)
drop4 = Dropout(0.3)(dense4)
dense5 = Dense(20,activation='relu')(drop4)
drop5= Dropout(0.3)(dense5)
dense6 = Dense(10,activation='relu')(drop5)
drop6 = Dropout(0.3)(dense6)
output1= Dense(1)(drop6)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor='val_loss', # 평가 지표로 확인하겠다
    mode= 'min', # 최대값 max, 알아서 찾아줘:auto
    patience=10, # 10번까지 초과해도 넘어가겠다
    restore_best_weights=True # val_loss값이 가장 낮은 값으로 저장 해놓겠다(False시 => 최소값 이후 10번째 값으로 그냥 잡는다.)
)
import datetime

hist = model.fit(x_train,y_train, epochs= 100, batch_size= 3,verbose=2,validation_split=0.1, callbacks=[es])

print(hist.history['loss'])
print(hist.history['val_loss'])

loss = model.evaluate(x_test,y_test)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교
r2 = r2_score(y_test, result)
print('loss :',loss)
print('R2 :',r2)

# loss : 2747.8154296875
# R2 : 0.6292789240425394