#https://dacon.io/competitions/official/236068/leaderboard

# 첫 분류 타겟값이 정해져 있다.
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

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

path = '.\_save\Keras28_mcp\\07_dacon_diabetes\\'
model = load_model(path+'0014-0.4955Keras28_MCP_save_07_dacondiabetes.hdf5')
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

#4. 평가, 예측
print('#############################')
results = model.evaluate(x_test, y_test)
print(results) 
#[0.034758370369672775, 0.9824561476707458]

#print('loss = ',results[0])
#print('acc = ', round(results[1],4)) # 반올림
y_predict = model.predict(x_test)
y_predict =  (y_predict > 0.5).astype(int)
from sklearn.metrics import accuracy_score # 이진만 받을 수 있다
accuracy_score = accuracy_score(y_test, y_predict)
