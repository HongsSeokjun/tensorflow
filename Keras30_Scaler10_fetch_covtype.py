from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from tensorflow.keras.models import Sequential,load_model
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
datasets = fetch_covtype()
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
x = datasets.data
y = datasets.target

#print(x.shape, y.shape) # (581012, 54) (581012,)
#print(np.unique(y, return_counts = True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
#print(pd.value_counts(y))

# encorder = OneHotEncoder(sparse=False)
# y = y.reshape(-1,1)
# y = encorder.fit_transform(y)

y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.05, random_state= 47)#,stratify=y)
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

path = '.\_save\Keras28_mcp\\10_fetch_covtype\\'
model = load_model(path+'0002-0.4911Keras28_MCP_save_10_fetch_covtype.hdf5')
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
#4. 평가,훈련
loss = model.evaluate(x_test, y_test)
print('loss :', loss[0])
print('acc :',loss[1])
y_predict = model.predict(x_test)

# loss : 10675.8515625
# acc : 0.03607448935508728
# MinMaxScaler
# loss : 1.9823029041290283
# acc : 0.49024128913879395
# StandardScaler
# loss : 0.4877546429634094
# acc : 0.7962548732757568
# MaxAbsScaler
# loss : 2.3258121013641357
# acc : 0.36291348934173584
# RobustScaler
# loss : 0.896658182144165
# acc : 0.6801831126213074
