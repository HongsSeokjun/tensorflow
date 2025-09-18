import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, BatchNormalization

a = np.array(range(1,101))
x_predict = np.array(range(96,106)) # 101 과 106을 찾기
#(96~ 106) => 결과로 (101~ 107)
# print(a.shape) (100,)
print(x_predict) #(10,) [ 96  97  98  99 100 101 102 103 104 105]

timesteps = 6

def split_xy(dataset, timesteps):
    x, y = [], []
    for i in range(len(dataset) - timesteps+1):
        x_window = dataset[i : i + timesteps-1]
        y_label = dataset[i + timesteps-1]
        x.append(x_window)
        y.append(y_label)
    return np.array(x), np.array(y)

def split_xy_predict(dataset, timesteps):
    x = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i:i + timesteps]
        x.append(subset)
    return np.array(x)


bbb = split_xy_predict(a,timesteps)
x_pred = split_xy_predict(x_predict, timesteps=5)
x = bbb[:,:-1]
y = bbb[:,5]
print(x)
print(x.shape) #=> (95, 5)
print(y)
print(y.shape)#(95,)
exit()
#2. 모델구성
model = Sequential()
model.add(SimpleRNN(100, input_shape=(5,1), activation='relu')) # return_sequences=True 차원 유지
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs= 300, shuffle=False,validation_split=0.1)

#4. 평가 예측
result = model.evaluate(x,y)
print('loss :', result)

x_pred = x_pred.reshape(-1, 5, 1)
# print(x_pred)
y_pred = model.predict(x_pred)
#print(y_pred.shape)
print("예측값 (101~107):", y_pred.flatten())
# loss : 0.030867762863636017
# 예측값 (101~106):
# [[101.16506]
#  [102.17056]
#  [103.17605]
#  [104.18156]
#  [105.18705]
#  [106.19254]]