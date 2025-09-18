import sklearn as sk
print(sk.__version__) #0.24.2 #1.1.3
from sklearn.datasets import load_boston
import numpy as np
from keras.models import Sequential, load_model # 모델 불러오기
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
from sklearn.preprocessing import StandardScaler,MinMaxScaler

#1. 데이터
dataset = load_boston()
#print(dataset)
#print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

x = dataset.data
y = dataset.target
#print(x.shape, y.shape) #(506, 13) (506,)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state=42)

scaler = StandardScaler()
minmaxs = MinMaxScaler()
x_train = minmaxs.fit_transform(x_train)
x_test = minmaxs.transform(x_test)

# print(np.min(x_train), np.max(x_train))
# print(np.min(x_test), np.max(x_test))

#2. 모델 구성

# model = Sequential([
#     Dense(32, input_dim = 13, activation='relu'),
#     Dense(16, activation='relu'),
#     Dense(8, activation='relu'),
#     Dense(4, activation='relu'),
#     Dense(1)
# ])
path = '.\_save\Keras26\\'
model = load_model(path+'keras26_1_save.h5')
# 아직 가중치를 가져오지는 못했다.

model.summary()
#model.save(path+ 'keras26_1_save.h5')

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss', # 평가 지표로 확인하겠다
    mode= 'min', # 최대값 max, 알아서 찾아줘:auto
    patience=10, # 10번까지 초과해도 넘어가겠다
    restore_best_weights=False # val_loss값이 가장 낮은 값으로 저장 해놓겠다(False시 => 최소값 이후 10번째 값으로 그냥 잡는다.)
)

hist = model.fit(x_train,y_train, epochs= 100, batch_size= 3,verbose=2,validation_split=0.1, callbacks=[es])
# 리스트의 형태로 디버그 값 (return loss, return val_loss)
print('======================== hist ====================')
print(hist)
print('======================== hist.history ====================')
print(hist.history)

#dictionary
#{'loss': [112.74440002441406, 77.60723114013672, 78.3528823852539, 73.68932342529297, 73.3847427368164, 72.53368377685547, 59.78982925415039, 68.60456085205078, 53.34999465942383, 60.53734588623047], 
#'val_loss': [46.672096252441406, 70.20635223388672, 42.35757827758789, 66.6172866821289, 59.02158737182617, 43.768577575683594, 78.94567108154297, 38.48080825805664, 38.67212677001953, 26.917438507080078]}

print(hist.history['loss'])
print(hist.history['val_loss'])

loss = model.evaluate(x_test,y_test)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교
r2 = r2_score(y_test, result)
print('loss :',loss)
print('R2 :',r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test,result)
print('Rmse :',rmse)

# loss : 7.98152494430542
# R2 : 0.8721610313148815
# Rmse : 2.8251594627113046