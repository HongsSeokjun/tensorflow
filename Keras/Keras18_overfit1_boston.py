# 과적합
# Keras17_val1_boston

import sklearn as sk
print(sk.__version__) #1.6.1 => 1.1.3

from sklearn.datasets import load_boston
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터

dataset = load_boston()
print(dataset)
print(dataset.DESCR)
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2, random_state=6514) #6514


#print(x)
# print(x.shape) #(506, 13)
# print(y)
# print(y.shape) #(506,)
#exit()
#2. 모델 구성
model = Sequential()
model.add(Dense(300, input_dim = 13, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train,y_train, epochs= 100, batch_size= 3,verbose=2,validation_split=0.2)
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

import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

plt.figure(figsize=(9,6)) # 9 x 6
plt.plot(hist.history['loss'], c = 'red', label = 'loss') # x축은 epochs, y값만 넣으면 시간순으로 그림 그림
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss')
plt.title('보스턴 Loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right') # 우측 상단에 라벨 표시
plt.grid() # 격자 표시
plt.show()

















'''

#4. 평가, 예측
print('#############################')
loss = model.evaluate(x_test,y_test)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교
print('loss :',loss)
#print('x의 예측값 :',result)

from sklearn.metrics import mean_squared_error,r2_score

# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))

# rmse = RMSE(y_test,result)
# print('Rmse :',rmse)
print('R2 :',r2_score(y_test,result))
# val 안넣은값
# loss : 23.180940628051758
# R2 : 0.7532197300733097

# val 넣은값
# loss : 25.35736656188965
# R2 : 0.7598389824774822

#프롬프트 엔지니어링 LLM 더 정확한 답을 이끌기 위한 작업
#하이퍼파라미터 튜닝 모델구성 이나 전처리 내용을 바꾸는거

'''