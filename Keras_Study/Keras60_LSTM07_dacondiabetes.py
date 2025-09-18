from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization,LSTM,Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score#불균형 데이터일때
from sklearn.preprocessing import StandardScaler,LabelEncoder
import tensorflow as tf
import matplotlib.pylab as plt
from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint
#1. 데이터
path='C:\study25ju\_data\dacon\당뇨병\\'
train_csv = pd.read_csv(path+'train.csv', index_col=0)
#print(train_csv) # [652 rows x 9 columns]

test_csv = pd.read_csv(path+'test.csv', index_col=0)
#print(test_csv) #[116 rows x 8 columns]

submission_csv = pd.read_csv(path+'sample_submission.csv', index_col=0)
#print(submission_csv) #[116 rows x 1 columns]

x = train_csv.drop(['Outcome'], axis= 1)

x = x.replace(0, np.nan)
x = x.fillna(train_csv.min())
# x= x.dropna()
#print(x)
#exit()
y = train_csv['Outcome']
print(x.shape)#(652, 8)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state= 47)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 8, 1)  # (batch, height, width, channel)
x_test = x_test.reshape(-1, 8, 1)  # (batch, height, width, channel)



#2. 모델구성
model = Sequential()
model.add(LSTM(32,input_shape=(8,1), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
# model.add(Flatten())
model.add(Dense(8,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(
    monitor='val_loss', # 평가 지표로 확인하겠다
    mode= 'min', # 최대값 max, 알아서 찾아줘:auto
    patience=10, # 10번까지 초과해도 넘어가겠다
    restore_best_weights= True # val_loss값이 가장 낮은 값으로 저장 해놓겠다(False시 => 최소값 이후 10번째 값으로 그냥 잡는다.)
)
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train,epochs= 100, batch_size= 4,verbose=2,validation_split=0.1,callbacks=[es])


# 4. 평가 예측

loss = model.evaluate(x_test,y_test)

# r2 = r2_score(y_test, result)
print('loss :',loss)
y_predict = model.predict(x_test)
y_predict =  (y_predict > 0.5).astype(int)
from sklearn.metrics import accuracy_score # 이진만 받을 수 있다
accuracy_score = accuracy_score(y_test, y_predict)
print('acc',accuracy_score)
# loss : [0.42678943276405334, 0.8030303120613098]
# acc 0.803030303030303

# acc 0.7424242424242424