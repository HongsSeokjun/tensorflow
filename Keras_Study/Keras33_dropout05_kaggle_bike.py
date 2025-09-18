#https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np #import numpy as np
import pandas as pd # import pandas as pd
from tensorflow.keras.models import Sequential # from tensorflow.python.keras.models
from tensorflow.keras.layers import Dense, Dropout #from tensorflow.
from sklearn.model_selection import train_test_split # sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error # sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터

path = './_data/Kaggle/bike/' # 상대경로

train_csv = pd.read_csv(path+'train.csv', index_col=0) #train_csv = pd.read_csv(path+'train.csv')
test_csv = pd.read_csv(path+'test.csv', index_col=0) #test_csv = pd.read_csv(path+'test.csv')
submission_csv = pd.read_csv(path+'sampleSubmission.csv') # submission_csv = pd.read_csv(path+'sampleSubmission.csv')


x = train_csv.drop(['casual','registered','count'], axis=1) #'casual','registered','count'
#print(x) #[10886 rows x 8 columns] x = train_csv.drop(['casual','registered','count',], axis=1) 리스트에 , 추가로 찍어도 오류가 안난다
y = train_csv['count']
#print(y)
#print(y.shape) #(10886,) pandas 데이터형태 serise(백터), dataframe(행렬) 2가지 형태
#exit()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.08, random_state=117) #117

# 2. 모델 구성
model = Sequential()
model.add(Dense(240, activation='relu', input_dim =8))
model.add(Dropout(0.3))
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))
#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(
    monitor='val_loss', # 평가 지표로 확인하겠다
    mode= 'min', # 최대값 max, 알아서 찾아줘:auto
    patience=20, # 10번까지 초과해도 넘어가겠다
    restore_best_weights=True # val_loss값이 가장 낮은 값으로 저장 해놓겠다(False시 => 최소값 이후 10번째 값으로 그냥 잡는다.)
)
import datetime
date = datetime.datetime.now()
print(date)
print(type(date))
date = date.strftime('%m%d_%H%M%S')
print(date)
print(type(date)) # <class 'str'>

hist = model.fit(x_train,y_train, epochs= 100, batch_size= 20,verbose=2,validation_split=0.05, callbacks=[es])

#print(hist.history['loss'])
#print(hist.history['val_loss'])

loss = model.evaluate(x_test,y_test)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교
r2 = r2_score(y_test, result)
print('loss :',loss)
print('R2 :',r2)

# loss : 22662.279296875
# R2 : 0.3075740784792408