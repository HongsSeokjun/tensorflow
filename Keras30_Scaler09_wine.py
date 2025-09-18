from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential,load_model
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score#불균형 데이터일때
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

#print(x.shape, y.shape) #(178, 13) (178,)
#print(np.unique(y,return_counts = True))
# #(array([0, 1, 2]), array([59, 71, 48], dtype=int64))
#print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48
############### OneHotEncoding (반드시 y에서만)###########
encorder = OneHotEncoder(sparse=False)
y = y.reshape(-1,1)
y = encorder.fit_transform(y)
#y = pd.get_dummies(y)
# print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.05, random_state= 42,stratify=y)
# stratify=y 전략 0,1,2의 열을 정확하게 나눠서 골고루 섞게 만들어준다.
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
labels = np.argmax(y_train, axis=1)
print(labels)
#labels1 = np.argmax(y_test, axis=1)
print(np.unique(labels, return_counts=True))

path = '.\_save\Keras28_mcp\\09_wine\\'
model = load_model(path+'0116-0.3376Keras28_MCP_save_09_wine.hdf5')
#3. 컴파일, 훈련
#model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
#4. 평가,훈련
loss = model.evaluate(x_test, y_test)
print('loss :', loss[0])
print('acc :',loss[1])
y_predict = model.predict(x_test)

# loss : 0.23308637738227844
# acc : 1.0
#MinMaxScaler
# loss : 7.505106449127197
# acc : 0.4444444477558136
#StandardScaler
# loss : 16.23764991760254
# acc : 0.4444444477558136
#MaxAbsScaler
# loss : 10.4655122756958
# acc : 0.4444444477558136
#RobustScaler
# loss : 10.805785179138184
# acc : 0.4444444477558136