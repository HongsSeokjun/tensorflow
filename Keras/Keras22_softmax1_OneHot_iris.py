#다중 분류 3가지 꽃/softmax
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import time

#1. 데이터
datasets = load_iris()

x= datasets.data
y= datasets.target
#print(x.shape, y.shape)#(150, 4), (150,)
#print(y)
#print(np.unique(y, return_counts=True))
#(array([0, 1, 2,]), array([50, 50, 50], dtype=int64))
#print(pd.DataFrame(y).value_counts())
#print(pd.value_counts(y))
# 0    50
# 1    50
# 2    50
# 다중 분류에서는 반드시 OneHotEncoding => 자리 0,1,2가 동일한 값을 가지고 있기 때문에
# LabelEncoder => x만 처리 했었다
############### OneHotEncoding (반드시 y에서만)###########

#1. sklearn용 #입력	2D 배열 (범주형 문자열 or 정수)	1D 정수 배열 (클래스 인덱스)
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)  # metrics로 반환 받기 때문에 N,1로 reshape하고 해야한다.
y = y.reshape(-1, 1)
# -1은 "행 개수는 내가 알아서 맞출게"
# 1은 "열은 1개짜리로 만들어줘"
y = encoder.fit_transform(y)
#y=y.toarray() #사이파이를 넘파이 형태로 변환 싸이파이 => 희소행렬방식

#2. pd용
#df = pd.DataFrame({'label': y})
# One-Hot 인코딩 → 결과에서 column 없이 숫자만
#y = pd.get_dummies(df['label']).values 
#y = pd.get_dummies(y)
#print(y_encoded.shape)#(150, 3)
#3. keras용 => 처음 시작하는 값이 0이 없다면 임의로 0칼람을 만들어 버린다. 결국 .shape 찍어봐야 함.
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)


#print(y)
#print(y.shape[1])
#######################################################
x_train, x_test, y_train,_y_test = train_test_split(x,y,test_size=0.1,random_state=478)
#2. 모델 구성
model = Sequential([
    Dense(64, input_dim = 4,activation='relu'),
    (Dropout(0.1)),
    Dense(32, activation='relu'),
    (Dropout(0.1)),
    Dense(8, activation='relu'),
    (Dropout(0.1)),
    Dense(y.shape[1], activation='softmax')] # 일반 회귀 모델은 실수를 출력하고 있다
)


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

#다중 클래스 분류	Dense(클래스 수, softmax)	categorical_crossentropy	OneHotEncoded (ex: [1,0,0])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True
)
start_time = time.time
model.fit(x_train, y_train, epochs= 300,batch_size= 3,verbose=2 ,validation_split= 0.1,callbacks=[es])
end_time = time.time
#4. 평가 훈련

loss = model.evaluate(x_test, _y_test)
print('loss :', loss[0])
print('acc :', loss[1])
#print(round(end_time- start_time,2))