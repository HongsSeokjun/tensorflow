import numpy as np
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
#1. 데이터
dataset = load_diabetes()

x = dataset.data
y = dataset.target

#print(x)
#print(y)
#print(x.shape) #(442, 10)
#print(y.shape) #(442,)
#exit()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=947)#947
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
path = '.\_save\Keras28_mcp\\03_diabetes\\'
model = load_model(path+'0031-2589.3835Keras28_MCP_save_03_diabetes.hdf5')
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

#4. 평가, 예측
print('#############################')
result = model.predict(x_test)
print('result :',result)
r2 = r2_score(y_test, result)
print('R2 :',r2)
# R2 : 0.5895985510462973
#MinMaxScaler
#R2 : -85.80164216061465
#MaxAbsScaler
#R2 : -218.45993602805666
#MaxAbsScaler
#R2 : -29.78203872194488
#RobustScaler
#R2 : -101.20774337243965
