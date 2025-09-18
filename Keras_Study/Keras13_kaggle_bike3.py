# train.csv와 new_test.csv로 count 예측

import numpy as np 
import pandas as pd 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터
path = './_data/Kaggle/bike/'

train_csv = pd.read_csv(path+'train.csv', index_col=0) #train_csv = pd.read_csv(path+'train.csv')
test_csv = pd.read_csv(path+'new_test.csv') #test_csv = pd.read_csv(path+'test.csv')
submission_csv = pd.read_csv(path+'sampleSubmission.csv') 

x = train_csv.drop(['count'], axis=1) #'casual','registered','count'
#print(x) #[10886 rows x 8 columns] x = train_csv.drop(['casual','registered','count',], axis=1) 리스트에 , 추가로 찍어도 오류가 안난다
y = train_csv['count']

# print(x.shape)
# print(y.shape)
# exit()
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.1, random_state= 548) #117

# 2. 모델 구성
model = Sequential()
model.add(Dense(80, activation='relu', input_dim =10))
model.add(Dense(20, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs= 100, batch_size= 20)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
result = model.predict(x_test)

print('loss :', loss)
def rmse(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print('Rmse :', rmse(y_test, result))
print('R2 :', r2_score(y_test, result))

############### CSV 파일 만들기 ###############
# y_submit = model.predict(test_csv)
# submission_csv['count'] = y_submit
# submission_csv.to_csv(path+'submission_0522_1700.csv', index=False)


# trian['datatime'] = pd.to_datetime()# Convert datetime
# train['datetime'] = pd.to_datetime(train['datetime'])
# test['datetime'] = pd.to_datetime(test['datetime'])
# train['hour'] = train['datetime'].dt.hour
# train['day'] = train['datetime'].dt.dayofweek
# train['month'] = train['datetime'].dt.month
# train['year'] = train['datetime'].dt.year.map({2011:0, 2012:1})



#df['is_weekend'] = df['day'].apply(is_weekend)


# def is_weekend(x):
#     if x >= 5:
#         return 1
#     else:
#         return 0