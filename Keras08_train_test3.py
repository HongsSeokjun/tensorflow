import numpy as np
from tensorflow.keras.models import Sequential #대문자면 대부분 클래스
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split #소문자면 대부분 함수
#사이킷런 임포트

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
# print(x.shape) #(10,)
# print(y.shape) #(10,)

#[실습] 넘파이 리스트의 슬라이싱 데이터 전처리
#print(a[3:]) #[4,5]
#print(x[0:-5]) #[2 3 4]
#print(x[0:10:3]) #[1 3] 2는 간격 (step) → 인덱스를 2씩 건너뜀  과적합을 막으려고 훈련과 테스트를 나눠서 섞는것
x_train, x_test, y_train, y_test =train_test_split(x, y, train_size=0.7,
                                    test_size=0.3, shuffle =True,
                                    random_state=121) 

#(test_size default = 0.25) (shuffle  default = True)

#x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)
print("훈련 데이터 x:", x_train)
print("테스트 데이터 x:", x_test)
print("훈련 데이터 y:", y_train)
print("테스트 데이터 y:", y_test)


#exit()

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