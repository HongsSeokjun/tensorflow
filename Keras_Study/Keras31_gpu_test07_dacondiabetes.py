#https://dacon.io/competitions/official/236068/leaderboard

# 첫 분류 타겟값이 정해져 있다.
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import time
import tensorflow as tf
# 1. 데이터

path='./_data/dacon/diabetes/'
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

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.1, random_state= 46)

#2. 모델 구조

model = Sequential()
model.add(Dense(64, input_dim = 8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 이진 분류는 무조건 마지막 Dense sigmoid, 0.5기준으로 0과 1로 만들기 

#3. 컴파일, 훈련

#model.compile(loss='mse', optimizer='adam')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # 이진 분류 loss는 무조건
start = time.time()
hist = model.fit(x_train, y_train,epochs= 100, batch_size= 32,verbose=2,validation_split=0.1)#,class_weight=class_weights,)
end = time.time()
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    print('GPU 있다')
else:
    print('GPU 없다')


print("걸린시간 :",end-start)
# GPU 있다
# 걸린시간 : 8.215262174606323
# GPU 없다
# 걸린시간 : 3.1673026084899902