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
                                                    test_size=0.25, random_state=6514)


#print(x)
# print(x.shape) #(506, 13)
# print(y)
# print(y.shape) #(506,)
#exit()
#2. 모델 구성
model = Sequential()
model.add(Dense(300, input_dim = 13))
model.add(Dense(400))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs= 400, batch_size= 3)

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
# loss : 23.180940628051758
# R2 : 0.7532197300733097

#프롬프트 엔지니어링 LLM 더 정확한 답을 이끌기 위한 작업
#하이퍼파라미터 튜닝 모델구성 이나 전처리 내용을 바꾸는거