from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score#불균형 데이터일때
from sklearn.preprocessing import OneHotEncoder
#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

#print(x.shape, y.shape) #(178, 13) (178,)
#print(np.unique(y,return_counts = True))
# #(array([0, 1, 2]), array([59, 71, 48], dtype=int64))
#print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48
############### OneHotEncoding (반드시 y에서만)###########
encorder = OneHotEncoder(sparse=False)
y = y.reshape(-1,1)
y = encorder.fit_transform(y)
#y = pd.get_dummies(y)
# print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.05, random_state= 42,stratify=y)
# stratify=y 전략 0,1,2의 열을 정확하게 나눠서 골고루 섞게 만들어준다.
labels = np.argmax(y_train, axis=1)
print(labels)
#labels1 = np.argmax(y_test, axis=1)
print(np.unique(labels, return_counts=True))

exit()
#2. 모델구성
model = Sequential([
 Dense(32, input_dim = 13,activation='relu'),
 BatchNormalization(),
 (Dropout(0.3)),
 Dense(16, activation='relu'),
 BatchNormalization(),
 (Dropout(0.3)),
 Dense(y.shape[1], activation='softmax')]
)
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True,
)
start_time = time.time
model.fit(x_train,y_train,epochs=300, batch_size=3,verbose=2,validation_split=0.1,callbacks=[es])
end_time = time.time
#4. 평가,훈련
loss = model.evaluate(x_test, y_test)
print('loss :', loss[0])
print('acc :',loss[1])
y_predict = model.predict(x_test)
#print(y_predict)
# y_predict =  (y_predict > 0.5).astype(int)
# f1_score1 = f1_score(y_test,y_predict, average='macro')
# print(f1_score1)