
import numpy as np #import numpy as np
import pandas as pd # import pandas as pd
from tensorflow.keras.models import Sequential,load_model # from tensorflow.python.keras.models
from tensorflow.keras.layers import Dense #from tensorflow.
from sklearn.model_selection import train_test_split # sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error # sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터

path = './_data/Kaggle/bike/' # 상대경로

train_csv = pd.read_csv(path+'train.csv', index_col=0) #train_csv = pd.read_csv(path+'train.csv')
test_csv = pd.read_csv(path+'test.csv', index_col=0) #test_csv = pd.read_csv(path+'test.csv')
submission_csv = pd.read_csv(path+'sampleSubmission.csv') # submission_csv = pd.read_csv(path+'sampleSubmission.csv')


x = train_csv.drop(['casual','registered','count'], axis=1) #'casual','registered','count'
#print(x) #[10886 rows x 8 columns] x = train_csv.drop(['casual','registered','count',], axis=1) 리스트에 , 추가로 찍어도 오류가 안난다
y = train_csv['count']
#print(y)
#print(y.shape) #(10886,) pandas 데이터형태 serise(백터), dataframe(행렬) 2가지 형태
#exit()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.08, random_state=117) #117


path = '.\_save\Keras28_mcp\\05_kaggle_bike\\'
model = load_model(path+'0094-23393.0859Keras28_MCP_save_08_kaggle_bank.hdf5')
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

#4. 평가, 예측
print('#############################')
result = model.predict(x_test)
print('result :',result)
r2 = r2_score(y_test, result)
print('R2 :',r2)
