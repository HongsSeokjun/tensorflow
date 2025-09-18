#https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np #import numpy as np
import pandas as pd # import pandas as pd
from tensorflow.keras.models import Sequential # from tensorflow.python.keras.models
from tensorflow.keras.layers import Dense #from tensorflow.
from sklearn.model_selection import train_test_split # sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error # sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터

path = './_data/Kaggle/bike/' # 상대경로
#path = 'c:/Study25/_data/Kaggle/bike/' # 절대 경로 (경로 전체)

train_csv = pd.read_csv(path+'train.csv', index_col=0) #train_csv = pd.read_csv(path+'train.csv')
test_csv = pd.read_csv(path+'test.csv', index_col=0) #test_csv = pd.read_csv(path+'test.csv')

######### x와 y 분리 ##########
#x = train_csv.drop(['casual','registered','count'], axis=1)
x = train_csv.drop(['casual','registered','count'], axis=1) #'casual','registered' => 뿐만아니라 여러 컬럼을 만드는걸 파생 컬럼 이라고 부름
#print(x) #[10886 rows x 8 columns] x = train_csv.drop(['casual','registered','count',], axis=1) 리스트에 , 추가로 찍어도 오류가 안난다
y = train_csv[['casual','registered']]
#print(y)
#print(y.shape) #(10886,) pandas 데이터형태 serise(백터), dataframe(행렬) 2가지 형태
#exit()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.2, random_state= 1197) #117

# 2. 모델 구성
model = Sequential()
model.add(Dense(240, activation='relu', input_dim =8))
model.add(Dense(120, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs= 1, batch_size= 32)

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

print('test_csv 타입 :', type(test_csv)) #  <class 'pandas.core.frame.DataFrame'>
print('y_submit 타입 :', type(y_submit)) #  <class 'numpy.ndarray'>
# 기본적으로 넘파이로 구성 => 자동으로 바꿔준다.

test2_csv = test_csv # 원래는 .copy()를 사용해야함

test2_csv = pd.DataFrame(y_submit, ['casual','registered'])

test2_csv[['casual','registered']] = y_submit

test2_csv.to_csv(path+'new_test.csv', index=False)

# def feature_engineering(df):
#     df['datetime'] = pd.to_datetime(df['datetime'])
#     df['hour'] = df['datetime'].dt.hour()
#     df['day'] = df['day'].dt.dayofweek
#     df['month'] = df['datetime'].dt.month
#     df['year'] = df['datetime'].dt.year.map({2011:0, 2012: 1})
#     df['is_weekend'] = df['day'].apply(lambda x:  )
    
    