import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
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
model = Sequential()
model.add(Dense(300, input_dim = 10, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss', # 평가 지표로 확인하겠다
    mode= 'min', # 최대값 max, 알아서 찾아줘:auto
    patience=10, # 10번까지 초과해도 넘어가겠다
    restore_best_weights=True # val_loss값이 가장 낮은 값으로 저장 해놓겠다(False시 => 최소값 이후 10번째 값으로 그냥 잡는다.)
)


hist = model.fit(x_train,y_train, epochs= 100, batch_size= 3,verbose=2,validation_split=0.1, callbacks=[es])

print(hist.history['loss'])
print(hist.history['val_loss'])

loss = model.evaluate(x_test,y_test)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교
r2 = r2_score(y_test, result)
print('loss :',loss)
print('R2 :',r2)

#print('x의 예측값 :',result)

# default Epoch 100
# loss : 4171.52490234375
# R2 : 0.4371994901719861
#val_loss2825.64306640625

# EarlyStopping, restore_best_weights=True, patience=10 Epoch 52/100
# loss : 3037.63623046875
# R2 : 0.5901778372433126
# val_loss 2525.943359375

# EarlyStopping, restore_best_weights=False, patience=10  Epoch 45/100
# loss : 2648.282470703125
#R2 : 0.642707469708341
# val_loss 2867.529541015625