from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization,Conv1D,Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score#불균형 데이터일때
from sklearn.preprocessing import StandardScaler,LabelEncoder
import tensorflow as tf
import matplotlib.pylab as plt
from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint
#1. 데이터
path ='C:\study25ju\_data\kaggle\otto\\'

train_csv = pd.read_csv(path+'train.csv',index_col=0)
test_csv =pd.read_csv(path+'test.csv', index_col=0)
submission_csv = pd.read_csv(path+'sampleSubmission.csv',index_col=0)

x = train_csv.drop(['target'],axis=1)
y = train_csv['target']
#print(np.unique(y, return_counts = True))
#(array(['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6',
#       'Class_7', 'Class_8', 'Class_9'], dtype=object), array([ 1929, 16122,  8004,  2691,  2739, 14135,  2839,  8464,  4955],
y = pd.get_dummies(y)
 
print(x.shape) #(61878, 93)
print(y.shape) #(61878, 9)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5, random_state= 47)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 93, 1)  # (batch, height, width, channel)
x_test = x_test.reshape(-1, 93, 1)  # (batch, height, width, channel)


#2. 모델구성
model = Sequential()
model.add(Conv1D(32,kernel_size=2,input_shape=(93,1), activation='relu'))
model.add(Conv1D(30,2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(8,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(
    monitor='val_loss', # 평가 지표로 확인하겠다
    mode= 'min', # 최대값 max, 알아서 찾아줘:auto
    patience=10, # 10번까지 초과해도 넘어가겠다
    restore_best_weights= True # val_loss값이 가장 낮은 값으로 저장 해놓겠다(False시 => 최소값 이후 10번째 값으로 그냥 잡는다.)
)
model.compile(loss='categorical_crossentropy',optimizer='adam')
hist = model.fit(x_train, y_train,epochs= 1, batch_size= 32,verbose=2,validation_split=0.1,callbacks=[es])


# 4. 평가 예측

loss = model.evaluate(x_test,y_test)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교

r2 = r2_score(y_test, result)
print('loss :',loss)
print('result :',result)
print('r2 :',r2)
# loss : 0.7068373560905457
# r2 : 0.5209084461493635