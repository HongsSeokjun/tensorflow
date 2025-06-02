#https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np #import numpy as np
import pandas as pd # import pandas as pd
from tensorflow.python.keras.models import Sequential # from tensorflow.python.keras.models
from tensorflow.python.keras.layers import Dense #from tensorflow.
from sklearn.model_selection import train_test_split # sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error # sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터

path = './_data/Kaggle/bike/' # 상대경로
#path = '.\_data\kaggle\\bike\\' #\n 줄 바꿈 \a , \b등 예약된 놈들빼고는 다 된다.
#path = '.\\_data\kaggle\bike\\'
#path = './/_data//kaggle//bike//'

path = 'c:/Study25/_data/Kaggle/bike/' # 절대 경로 (경로 전체)

train_csv = pd.read_csv(path+'train.csv', index_col=0) #train_csv = pd.read_csv(path+'train.csv')
test_csv = pd.read_csv(path+'test.csv', index_col=0) #test_csv = pd.read_csv(path+'test.csv')
submission_csv = pd.read_csv(path+'sampleSubmission.csv', index_col=0) # submission_csv = pd.read_csv(path+'sampleSubmission.csv')

print(train_csv)
print(train_csv.shape) #(10886, 11)
print(test_csv.shape) #(6493, 8)
print(submission_csv.shape) #(6493, 1)

exit()

print(train_csv.info()) #non 값 없음
#print(test_csv.columns)
feature = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp','humidity', 'windspeed']
x = train_csv[feature]
print(x) #(10886, 10)
y = train_csv['count']
print(y.shape) #(10886,)
#print(test_csv.shape)
exit()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.1, random_state= 17)

# 2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim = 8, activation= 'relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs= 80, batch_size= 30)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
result = model.predict(x_test)

print('loss :', loss)
def rmse(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print('Rmse :', rmse(y_test, result))
print('R2 :', r2_score(y_test, result))

y_submit = model.predict(test_csv)
submission_csv['count'] = y_submit

# loss : 22470.63671875
# Rmse : 149.90209574085756
# R2 : 0.3528048262879112