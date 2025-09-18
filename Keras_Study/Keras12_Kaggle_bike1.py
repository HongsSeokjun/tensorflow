#https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np #import numpy as np
import pandas as pd # import pandas as pd
from tensorflow.keras.models import Sequential # from tensorflow.python.keras.models
from tensorflow.keras.layers import Dense #from tensorflow.
from sklearn.model_selection import train_test_split # sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error # sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터

path = './_data/Kaggle/bike/' # 상대경로
#path = '.\_data\kaggle\\bike\\' #\n 줄 바꿈 \a , \b등 예약된 놈들빼고는 다 된다.
#path = '.\\_data\kaggle\bike\\'
#path = './/_data//kaggle//bike//'

#path = 'c:/Study25/_data/Kaggle/bike/' # 절대 경로 (경로 전체)

train_csv = pd.read_csv(path+'train.csv', index_col=0) #train_csv = pd.read_csv(path+'train.csv')
test_csv = pd.read_csv(path+'test.csv', index_col=0) #test_csv = pd.read_csv(path+'test.csv')
submission_csv = pd.read_csv(path+'sampleSubmission.csv') # submission_csv = pd.read_csv(path+'sampleSubmission.csv')
#print(submission_csv.shape) #(6493, 2) index_col=0 => 날짜 데이터를 지웠다가, 다시 넣어주지 않으려고 처리 안함

#exit()
# print(train_csv)
# print(train_csv.shape) #(10886, 11)
# print(test_csv.shape) #(6493, 8)
# print(submission_csv.shape) #(6493, 1)

# print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='object')
# print(train_csv.info()) #non 값 없음
# print(train_csv.isnull().sum())  #non 값 없음
# print(train_csv.isna().sum())  #non 값 없음
# print(train_csv.describe())

#tmpsub = pd.read_csv(path+'submission_0522_1300.csv')
#tmpsub = tmpsub.iloc[:, 1:] #pandas// tmpsub = tmpsub.iloc[:, 1:] #numpy
# tmpsub.to_csv(path+'submission_0522_1300.csv', index=False)
# exit()

######### x와 y 분리 ##########

x = train_csv.drop(['casual','registered','count'], axis=1) #'casual','registered','count'
#print(x) #[10886 rows x 8 columns] x = train_csv.drop(['casual','registered','count',], axis=1) 리스트에 , 추가로 찍어도 오류가 안난다
y = train_csv['count']
#print(y)
#print(y.shape) #(10886,) pandas 데이터형태 serise(백터), dataframe(행렬) 2가지 형태
#exit()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.08, random_state= 1197) #117

# 2. 모델 구성
model = Sequential()
model.add(Dense(240, activation='relu', input_dim =8))
model.add(Dense(120, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs= 300, batch_size= 20)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
result = model.predict(x_test)

print('loss :', loss)
def rmse(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print('Rmse :', rmse(y_test, result))
print('R2 :', r2_score(y_test, result))

############### CSV 파일 만들기 ###############

y_submit = model.predict(test_csv)
#print(y_submit)
submission_csv['count'] = y_submit
#print(submission_csv) #count 칼럼에 다 넣어짐
submission_csv.to_csv(path+'submission_0522_1400.csv', index=False)

#ctrl + space => 자동완성

# loss : 22744.8828125
# Rmse : 150.81406849797722
# R2 : 0.41797369158513975