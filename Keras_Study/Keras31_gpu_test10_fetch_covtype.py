from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

datasets = fetch_covtype()
import tensorflow as tf
import time
x = datasets.data
y = datasets.target


#print(x.shape, y.shape) # (581012, 54) (581012,)
#print(np.unique(y, return_counts = True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
#print(pd.value_counts(y))

# encorder = OneHotEncoder(sparse=False)
# y = y.reshape(-1,1)
# y = encorder.fit_transform(y)

y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.05, random_state= 47)#,stratify=y)
#print(y_train.values)
#labels = np.argmax(y_train.values, axis=1)
# print(labels)
#print(np.unique(labels, return_counts=True))
standard = StandardScaler() #표준화
x_train= standard.fit_transform(x_train)        # train 데이터에 맞춰서 스케일링
x_test= standard.transform(x_test) # test 데이터는 transform만!

#2. 모델 구성
model = Sequential([
    Dense(256, input_dim=54, activation='relu'),
    BatchNormalization(),
    (Dropout(0.3)), #드랍아웃은 layer 깊이가 있을때 사용
    Dense(126, activation='relu'),
    BatchNormalization(),
    (Dropout(0.3)),
    Dense(62, activation='relu'),
    BatchNormalization(),
    (Dropout(0.3)),
    Dense(30, activation='relu'),
    BatchNormalization(),
    (Dropout(0.3)),
    Dense(16, activation='relu'),
    BatchNormalization(),
    (Dropout(0.3)),
    Dense(y.shape[1], activation='softmax')
])

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
start = time.time()
hist = model.fit(x_train, y_train,epochs= 20, batch_size= 32,verbose=2,validation_split=0.1)#,class_weight=class_weights,)
end = time.time()
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    print('GPU 있다')
else:
    print('GPU 없다')


print("걸린시간 :",end-start)