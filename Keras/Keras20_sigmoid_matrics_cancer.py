# 첫 분류 타겟값이 정해져 있다.
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer

#1. 데이터 Pandas 명령어는 외워라
datasets = load_breast_cancer()
#print(datasets.DESCR)
#print(datasets.feature_names) #<class 'sklearn.utils.Bunch'>
#print(type(datasets))
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']


x = datasets.data 
y = datasets.target
#print(x) 
#print(x.shape) #(569, 30) numpy
#print(y.shape) #(569,) numpy
# 회귀와 분류 => 지도학습(답안지가 있는 학습), 안지도 학습도 있다(주로 머신러닝)
# 분류 => 이진 분류, 다중 분류
# 분류 데이터일 경우 무조건 데이터가 불균형한지 확인해서 경우에 따라 데이터 증폭을 해야한다.
# 0과 1의 개수가 몇개인지 찾아보기.
#counts = np.bincount(y)
#print(counts) #[212 357]    

print(np.unique(y, return_counts=True))
# 1. 넘파이로 찾았을때
#(array([0, 1]), array([212, 357], dtype=int64))
#print(pd.value_counts(y))
# 1    357
# 0    212
#print(pd.DataFrame(y).value_counts())
# 1    357
# 0    212
#print(pd.Series(y).value_counts()) # 데이터가 하나로 치중되면 정확도가 무의미 해진다 => 1 = 56800000, 0 = 1 총 데이터의 개수가 중요
# 1    357
# 0    212

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.1, random_state= 7275)

#print(x_train.shape, x_test.shape) #(398, 30) (171, 30)
#print(y_train.shape, y_test.shape) #(398,) (171,)

#2. 모델 구조

model = Sequential()
model.add(Dense(64, input_dim = 30, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
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
start_time = time.time()
hist = model.fit(x_train, y_train,epochs= 10, batch_size= 6,verbose=2,validation_split=0.1, callbacks=[es])
end_time = time.time()
#4. 평가, 예측

results = model.evaluate(x_test, y_test)
print(results) 
#[0.034758370369672775, 0.9824561476707458]

print('loss = ',results[0])
print('acc = ', round(results[1],4)) # 반올림

y_predict = model.predict(x_test)
#print('y_predict :', y_predict)
print(y_predict[:10])
y_predict = np.round(y_predict)
print(y_predict[:10])

#exit()
#y_pred = np.array(y_predict) > 0.5   #y_pred = (y_predict > 0.5).astype(int) # 형변환

# 예측값이 실수(continuous)인데,
# 정답은 이진(binary)
from sklearn.metrics import accuracy_score # 이진만 받을 수 있다
accuracy_score = accuracy_score(y_test, y_predict)

print('acc_score :', accuracy_score)
print('걸린시간 :', round(end_time - start_time,2), '초')