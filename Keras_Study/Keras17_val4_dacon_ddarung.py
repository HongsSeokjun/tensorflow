# https://dacon.io/competitions/official/235576/data 최초의 null값 데이터

import numpy as np # 전처리
import pandas as pd # 전처리 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
#print(np.__version__)#1.23.0
#print(pd.__version__)#2.2.3

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
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs= 350, batch_size=20,validation_split=0.2)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
result = model.predict(x_test) # 정답 도출

print('loss :',loss)
def rmse(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print('Rmse :',rmse(y_test,result))
print('R2 :',r2_score(y_test, result))

import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

plt.figure(figsize=(9,6)) # 9 x 6
plt.plot(hist.history['loss'], c = 'red', label = 'loss') # x축은 epochs, y값만 넣으면 시간순으로 그림 그림
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss')
plt.title('따릉이 Loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right') # 우측 상단에 라벨 표시
plt.grid() # 격자 표시
plt.show()

# loss : 1602.4393310546875
# Rmse : 40.030481685837465
# R2 : 0.7872347534153281

# loss : 2240.453369140625
# Rmse : 47.33342823057056
# R2 : 0.6873394391381455

'''''
# submission.csv에 test_csv의 예측값 넣기
y_submit = model.predict(test_csv) # train데이터의 shape와 동일한 컬럼을 확인하고 넣어. x_train_shape:(N, 9)

#print('Rmse :',rmse(submission_,y_submit))
#print(y_submit.shape) # (715, 1)

######### submission.csv 파일 만들기 //count컬럼값만 넣어주기########
submission_csv['count'] = y_submit
#print(submission_csv)


##################### csv파일 만들기 #########################
submission_csv.to_csv(path + 'submission_0521_1530.csv') # CSV 만들기.
'''