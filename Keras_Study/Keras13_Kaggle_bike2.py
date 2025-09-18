# 1. train_csv 에서 casual과 registered 를 y로 잡는다.
# 2. 훈련해서, test_csv의 casual과 registered를 예측(preidct) 한다.
# 3. 예측한 casual과 registered를 test_csv에 컬럼으로 넣는다.
#   (N, 8) -> (N, 10) test.csv 파일을 new_test.csv 파일을 만든다.
#  

import numpy as np 
import pandas as pd 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_squared_error
# 1. 데이터
path = './_data/kaggle/bike/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)
test_csv = pd.read_csv(path+'test.csv', index_col=0)

######### x 와 y 분리
x = train_csv.drop(['casual','registered','count'], axis= 1)
y = train_csv[['casual','registered']]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.1, random_state= 1197)

# 2. 모델 구성
model = Sequential()
model.add(Dense(480, activation='relu', input_dim = 8))
model.add(Dense(200, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(2))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs= 200, batch_size=20)


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
result = model.predict(x_test)

print('R2 :', r2_score(y_test, result))

############### CSV 파일 만들기 ###############
y_submit = model.predict(test_csv)

test2_csv = test_csv # 원래는 .copy()를 사용해야함

#test2_csv = pd.DataFrame(y_submit, ['casual','registered'])

test2_csv[['casual','registered']] = y_submit

test2_csv.to_csv(path+'new_test.csv', index=False)