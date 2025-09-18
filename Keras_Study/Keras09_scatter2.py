import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.25, random_state=76542)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs= 300, batch_size= 2)
#4. 평가, 예측
print('#############################')
loss = model.evaluate(x_test,y_test)
result = model.predict([x])
print('loss :',loss)
print('x의 예측값 :',result)

# 그래프 그리기
import matplotlib.pylab as plt
plt.scatter(x,y,c='red') #데이터 점 찍기
plt.plot(x, result, color='green')
plt.show()

# loss : 24.75505256652832
# x의 예측값 : [[ 0.03371352]
#  [ 1.1276306 ]
#  [ 2.2215476 ]
#  [ 3.3154647 ]
#  [ 4.4093814 ]
#  [ 5.503298  ]
#  [ 6.597215  ]
#  [ 7.6911345 ]
#  [ 8.785048  ]
#  [ 9.878966  ]
#  [10.972887  ]
#  [12.066801  ]
#  [13.16072   ]
#  [14.254633  ]
#  [15.34855   ]
#  [16.442465  ]
#  [17.536385  ]
#  [18.630302  ]
#  [19.724224  ]
#  [20.818136  ]]
