import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
# print(x.shape) #(10,)
# print(y.shape) #(10,)

#[실습] 넘파이 리스트의 슬라이싱
# print(a[3:]) #[4,5]
# print(a[1:-1]) #[2 3 4]
# print(a[0:3:2]) #[1 3] 2는 간격 (step) → 인덱스를 2씩 건너뜀

x_train = x[0:7]
y_train = y[0:7]

x_test = x[7:10]
y_test = y[7:10]

print(x_train.shape, x_test.shape) #(7,) (3,)
print(y_train.shape, y_test.shape) #(7,) (3,) 과적합을 막으려고 훈련과 테스트를 나누는것
#훈련 데이터와 평가 데이터 => 성능 확인
# x_train = np.array([1,2,3,4,5,6,7])
# y_train = np.array([1,2,3,4,5,6,7])

# x_test = np.array([8,9,10])
# y_test = np.array([8,9,10])

exit()

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim = 1))
model.add(Dense(400))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss ='mse', optimizer ='adam')
model.fit(x_train,y_train,epochs =300, batch_size = 4)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
result = model.predict([11])

print('loss :', loss)
print('[11]의 예측값 :',result)
                       
# loss : 7.579122740649855e-14
# [11]의 예측값 : [[11.]]                      