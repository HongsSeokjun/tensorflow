#https://dacon.io/competitions/official/236068/leaderboard

# 첫 분류 타겟값이 정해져 있다.
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# 1. 데이터

path='./_data/dacon/diabetes/'
train_csv = pd.read_csv(path+'train.csv', index_col=0)
#print(train_csv) # [652 rows x 9 columns]

test_csv = pd.read_csv(path+'test.csv', index_col=0)
#print(test_csv) #[116 rows x 8 columns]

submission_csv = pd.read_csv(path+'sample_submission.csv', index_col=0)
#print(submission_csv) #[116 rows x 1 columns]

x = train_csv.drop(['Outcome'], axis= 1)

x = x.replace(0, np.nan)
x = x.fillna(train_csv.min())
# x= x.dropna()
#print(x)
#exit()
y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.1, random_state= 46)

#2. 모델 구조

model = Sequential()
model.add(Dense(64, input_dim = 8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 이진 분류는 무조건 마지막 Dense sigmoid, 0.5기준으로 0과 1로 만들기 

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss', # 평가 지표로 확인하겠다
    mode= 'min', # 최대값 max, 알아서 찾아줘:auto
    patience=30, # 10번까지 초과해도 넘어가겠다
    restore_best_weights=True # val_loss값이 가장 낮은 값으로 저장 해놓겠다(False시 => 최소값 이후 10번째 값으로 그냥 잡는다.)
)
#model.compile(loss='mse', optimizer='adam')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # 이진 분류 loss는 무조건
hist = model.fit(x_train, y_train,epochs= 20, batch_size= 3,verbose=2,validation_split=0.1, callbacks=[es])

#4. 평가, 예측

results = model.evaluate(x_test, y_test)
print(results) 
#[0.034758370369672775, 0.9824561476707458]

#print('loss = ',results[0])
#print('acc = ', round(results[1],4)) # 반올림
y_predict = model.predict(x_test)
y_predict =  (y_predict > 0.5).astype(int)
from sklearn.metrics import accuracy_score # 이진만 받을 수 있다
accuracy_score = accuracy_score(y_test, y_predict)



# submission.csv에 test_csv의 예측값 넣기
y_submit = model.predict(test_csv)
y_submit =  (y_submit > 0.5).astype(int)
#y_pred = [1 if y > 0.5 else 0 for y in y_submit]
######## submission.csv 파일 만들기 //count컬럼값만 넣어주기########
submission_csv['Outcome'] = y_submit
#print(submission_csv)


#################### csv파일 만들기 #########################
submission_csv.to_csv(path + 'submission_0527_1230.csv') # CSV 만들기.

import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

plt.figure(figsize=(9,6)) # 9 x 6
plt.plot(hist.history['loss'], c = 'red', label = 'loss') # x축은 epochs, y값만 넣으면 시간순으로 그림 그림
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss')
plt.title('당뇨 Loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right') # 우측 상단에 라벨 표시
plt.grid() # 격자 표시
plt.show()