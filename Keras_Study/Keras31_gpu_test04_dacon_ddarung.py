import numpy as np # 전처리
import pandas as pd # 전처리 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
#print(np.__version__)#1.23.0
#print(pd.__version__)#2.2.3
import time
import tensorflow as tf
# 1. 데이터

path='./_data/dacon/따릉이/'
train_csv = pd.read_csv(path+'train.csv', index_col=0)#. 현재 폴더 .. 이전 폴더
#print(train_csv) # [1459 rows x 11 columns] => [1459 rows x 10 columns]

test_csv = pd.read_csv(path+'test.csv', index_col=0)
#print(test_csv) #[715 rows x 9 columns]

submission_csv = pd.read_csv(path+'submission.csv', index_col=0)
#print(submission_csv) #[715 rows x 1 columns]

submission_ = pd.read_csv(path+'submission_0521_1400.csv',index_col=0)

#print(train_csv.shape) #(1459, 10)
#print(test_csv.shape) #(715, 9)
#print(submission_csv.shape) #(715, 1)

#print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object') Non- 결측치
#print(train_csv.info())

#print(train_csv.describe())
######################### 결측치 처리 1. 삭제 #############################
#print(train_csv.isnull().sum()) # 결측치의 개수 출력
#print(train_csv.isna().sum()) # 결측치의 개수 출력

train_csv = train_csv.dropna() # 결측치 제거 판다스 선처리 해야 함
#print(train_csv.isna().sum())
#print(train_csv.info())
#print(train_csv) #[1328 rows x 10 columns]

######################### 결측치 처리 2. 평균값 넣기 #############################

train_csv = train_csv.fillna(train_csv.mean()) #컬럼별 평균 [1459 rows x 9 columns]
# print(train_csv.isna().sum())
# print(train_csv.info())

######################### 테스트도 결측이 있다. #############################

#print(test_csv) 테이블이 밀릴 수 있어서 drpo 말고 mean으로 채워두기
test_csv = test_csv.fillna(test_csv.mean())
#print(test_csv.info())

x = train_csv.drop(['count'], axis= 1) # 행 또는 열 삭제
# count 라는 axis=1 열 삭제, 참고로 행 삭제는 axis = 0

y = train_csv['count']
#print(y.shape) #(1459,)

# features = ['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#             'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#             'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5']
# x = train_csv[features]

#print(x.hour.shape)
#exit()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state= 5917) #47 0.594 5917
# 2. 모델 구성
model = Sequential()
model.add(Dense(300, input_dim = 9, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
# 3,
model.compile(loss='mse', optimizer='adam')
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
# GPU 없다
# 걸린시간 : 5.397913217544556
# GPU 있다
# 걸린시간 : 11.707247257232666