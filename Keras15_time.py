# Keras14_verbose 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time # 시간에 대한 모듈 import

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])
# print(x.shape) #(10,)
# print(y.shape) #(10,)

#훈련 데이터와 평가 데이터 => 성능 확인
x_train = np.array(range(100))
y_train = np.array(range(100))

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

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
start_time = time.time() # 현재 시간을 반환, 시작시간.
model.fit(x_train,y_train,epochs =1000, batch_size = 256,
          verbose = 1) # 해당 훈련 과정이 다 디버그 찍히는게 비효율적일 수 있다.
end_time = time.time()
print('걸린시간 : ',end_time -start_time, '초')
#verbose = 0(결과만,침묵), 1(진행바, 디폴트), 2(진행바 없음), 3(에포만 나옴.) 

#1. 1000에포에서 0, 1, 2, 3의 시간을 적는다. batch_size = 4,
# 0= 18.0, 1= 24.2, 2= 20.17, 3= 21.8 (초)
#2. 1000에포에서 verbose = 1의 시간을 적는다.
# batch 1, 32, 128 일때 시간을 적는다.
# 1= 71.54, 32= 5.72, 128= 2.82 (초)

             