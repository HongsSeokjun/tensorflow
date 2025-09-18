from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization,Conv1D,Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score#불균형 데이터일때
from sklearn.preprocessing import StandardScaler,LabelEncoder
import tensorflow as tf
import matplotlib.pylab as plt
from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint
#1. 데이터
path = 'C:\study25ju\_data\kaggle\santander\\'

train_csv = pd.read_csv(path+'train.csv',index_col=0)
test_csv =pd.read_csv(path+'test.csv', index_col=0)
submission_csv = pd.read_csv(path+'sample_submission.csv',index_col=0)

print(train_csv)
#(200000, 201)
# Index(['target', 'var_0', 'var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6',
#        'var_7', 'var_8',
#        ...
#        'var_190', 'var_191', 'var_192', 'var_193', 'var_194', 'var_195',
#        'var_196', 'var_197', 'var_198', 'var_199'],
#       dtype='object', length=201)
x = train_csv.drop(['target'],axis=1)
y = train_csv['target']

print(x.shape) #(200000, 200)

x = x.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state= 47)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 50, 4)  # (batch, height, width, channel)
x_test = x_test.reshape(-1, 50, 4)  # (batch, height, width, channel)



#2. 모델구성
model = Sequential()
model.add(Conv1D(32,kernel_size=2,input_shape=(50,4), activation='relu'))
model.add(Conv1D(30,2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Flatten())
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
model.compile(loss='binary_crossentropy',optimizer='adam')
hist = model.fit(x_train, y_train,epochs= 1, batch_size= 32,verbose=2,validation_split=0.1,callbacks=[es])


# 4. 평가 예측

results = model.evaluate(x_test, y_test)
print(results)

y_predict = model.predict(x_test)
y_predict =  (y_predict > 0.5).astype(int)
f1_score1 = f1_score(y_test, y_predict)
print('f1_score :', f1_score1)
# 0.20584936439990997
# f1_score : 0.4900265957446809

# 0.2102176547050476
# f1_score : 0.46444520783236