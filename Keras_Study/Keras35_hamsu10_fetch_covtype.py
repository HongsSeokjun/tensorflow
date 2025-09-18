from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization,Input
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
datasets = fetch_covtype()

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
# model = Sequential([
#     Dense(256, input_dim=54, activation='relu'),
#     BatchNormalization(),
#     (Dropout(0.3)), #드랍아웃은 layer 깊이가 있을때 사용
#     Dense(126, activation='relu'),
#     BatchNormalization(),
#     (Dropout(0.3)),
#     Dense(62, activation='relu'),
#     BatchNormalization(),
#     (Dropout(0.3)),
#     Dense(30, activation='relu'),
#     BatchNormalization(),
#     (Dropout(0.3)),
#     Dense(16, activation='relu'),
#     BatchNormalization(),
#     (Dropout(0.3)),
#     Dense(y.shape[1], activation='softmax')
# ])

input1 = Input(shape=[54,]) # Sequential 모델의 input_shape랑 같음
dense1 = Dense(256, activation='relu')(input1) #ys1 summary에서 이름이 바뀜
Batch1 = BatchNormalization()(dense1)
drop1 = Dropout(0.3)(Batch1)
dense2 = Dense(126,activation='relu')(drop1)
Batch2 = BatchNormalization()(dense2)
drop2 = Dropout(0.3)(Batch2)
dense3 = Dense(62, activation='relu')(drop2) #ys1 summary에서 이름이 바뀜
Batch3 = BatchNormalization()(dense3)
drop3 = Dropout(0.3)(Batch3)
dense4 = Dense(30,activation='relu')(drop3)
Batch4 = BatchNormalization()(dense4)
drop4 = Dropout(0.3)(Batch4)
dense5 = Dense(16,activation='relu')(drop4)
Batch5 = BatchNormalization()(dense5)
drop5 = Dropout(0.3)(Batch5)
output1= Dense(y.shape[1], activation='softmax')(drop5)

model = Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련
labels = np.argmax(y_train.values, axis=1)
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels) ,
    y =labels
)
class_weights = dict(enumerate(weights))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True,
)
# import datetime
# date = datetime.datetime.now()
# print(date)
# print(type(date))
# date = date.strftime('%m%d_%H%M%S')
# print(date)
# print(type(date)) # <class 'str'>
# filename = '{epoch:04d}-{val_loss:.4f}Keras28_MCP_save_10_fetch_covtype.hdf5'
# path = '.\_save\Keras28_mcp\\10_fetch_covtype\\'
# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     save_best_only= True,
#     filepath=path+filename
# )

model.fit(x_train,y_train,epochs=200, batch_size=62,verbose=2,validation_split=0.1,callbacks=[es])#,class_weight=class_weights,)

#4. 평가,훈련
loss = model.evaluate(x_test, y_test)
print('loss :', loss[0])
print('acc :',loss[1])
y_predict = model.predict(x_test)

# loss : 0.39002886414527893
# acc : 0.8542218804359436

# loss : 0.4877546429634094
# acc : 0.7962548732757568