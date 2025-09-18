import numpy as np # 전처리
import pandas as pd # 전처리 
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
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
rain_csv = train_csv.dropna() # 결측치 제거 판다스 선처리 해야 함
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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=947)#947

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

path = '.\_save\Keras28_mcp\\04_dacon_ddarung\\'
model = load_model(path+'0038-3120.9233Keras28_MCP_save_04_dacon_ddarung.hdf5')
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

#4. 평가, 예측
print('#############################')
result = model.predict(x_test)
print('result :',result)
r2 = r2_score(y_test, result)
print('R2 :',r2)
#R2 : 0.6700948565904562
#MinMaxScaler
#R2 : -1.4963736913222716
#StandardScaler
#R2 : -1.3645955783521515
#MaxAbsScaler
#R2 : -1.4895817757095404
#RobustScaler
#R2 : -1.500394560018929