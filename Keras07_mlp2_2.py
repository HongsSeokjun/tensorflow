import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(10))
print(x) # [0 1 2 3 4 5 6 7 8 9]
print(x.shape) # (10,)

x = np.array(range(1, 10)) #[1 2 3 4 5 6 7 8 9]
print(x)
x = np.array(range(1,11)) #[ 1  2  3  4  5  6  7  8  9 10]
print(x)

x = np.array([range(10), range(21, 31), range(201 , 211)]) # 2개 이상의 데이터라 '[ ]' 필요
x = x.transpose()#x.T
print('aa',x.shape,'v') #(10, 3)
y = np.array([1,2,3,4,5,6,7,8,9,10])

#[실습]
#[10,31,211] 예측

#2. 모델 구성
model = Sequential()
model.add(Dense(10 , input_dim=3)) #딥러닝에서 노드를 1개로하면 데이터 손실이 있다 feature 데이터를 유지해야하는 이유
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss = 'mse',optimizer ='adam')
model.fit(x,y , epochs =100, batch_size=1)
#4. 평가, 예측

loss = model.evaluate(x,y)
print('loss :', loss)

result = model.predict([[10,31,211]])
print('result :', result)

#loss : 0.004348254296928644
# 1/1 [==============================] - 0s 74ms/step
# result : [[11.122934]]