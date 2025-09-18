# https://www.kaggle.com/competitions/playground-series-s4e1/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout,BatchNormalization,Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.utils import class_weight
import matplotlib.pylab as plt
import matplotlib
# 1. 데이터
path = './_data/Kaggle/bank/'

train_csv = pd.read_csv(path + 'train.csv',index_col=0)
test_csv = pd.read_csv(path + 'test.csv',index_col=0)
submission_csv = pd.read_csv(path+'sample_submission.csv',index_col=0)

#print(train_csv)#(165034, 13)
#print(train_csv.head(10)) # default 5개 => 원하는 값 위에서 개수까지 확인 가능
#print(train_csv.tail()) # default 5개 => 원하는 값 아래서 개수까지 확인 가능
#print(train_csv.isna().sum()) # 0
le_geo = LabelEncoder() # 클래스를 인스턴스 한다.
le_gender = LabelEncoder()
#print(train_csv.columns)
# Index(['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
#        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
#        'EstimatedSalary', 'Exited'],

train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le_gender.fit_transform(train_csv['Gender'])
print(train_csv['Geography'])
print(train_csv['Geography'].value_counts()) #pandas
# 0    94215
# 2    36213
# 1    34606
print(train_csv['Gender'].value_counts()) #pandas np.unique(data, return_counts=True)
# 1    93150
# 0    71884
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gender.transform(test_csv['Gender'])

train_csv = train_csv.drop(['CustomerId','Surname'],axis= 1)
test_csv = test_csv.drop(['CustomerId','Surname'], axis= 1)
print(train_csv.columns)
print(test_csv)
#exit()

#corr = train_csv.corr()  # 변수들 간 상관관계 계산
#plt.figure(figsize=(10,8))
#sns.boxplot(x=train_csv['Age'])
#sns.heatmap(corr, annot=True, cmap='coolwarm')  # annot=True는 숫자 표시
#plt.show()

#exit()
train_csv['Balance'] = train_csv['Balance'].replace(0, train_csv['Balance'].mean())
test_csv['Balance'] = test_csv['Balance'].replace(0, test_csv['Balance'].mean())
#train_csv.dropna()

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.1, random_state= 42)

standard = StandardScaler() # 표준화
scaler = MinMaxScaler() # 정규화
Robu =RobustScaler()
x_train[['CreditScore','Age','Tenure','Balance','EstimatedSalary']] = standard.fit_transform(x_train[['CreditScore','Age','Tenure','Balance','EstimatedSalary']])        # train 데이터에 맞춰서 스케일링
x_test[['CreditScore','Age','Tenure','Balance','EstimatedSalary']]= standard.transform(x_test[['CreditScore','Age','Tenure','Balance','EstimatedSalary']]) # test 데이터는 transform만!
test_csv[['CreditScore','Age','Tenure','Balance','EstimatedSalary']] = standard.transform(test_csv[['CreditScore','Age','Tenure','Balance','EstimatedSalary']])

# x_train[['EstimatedSalary']] = scaler.fit_transform(x_train[['EstimatedSalary']])        # train 데이터에 맞춰서 스케일링
# x_test[['EstimatedSalary']]= scaler.transform(x_test[['EstimatedSalary']]) # test 데이터는 transform만!
# test_csv[['EstimatedSalary']] = scaler.transform(test_csv[['EstimatedSalary']])

#7. Compute class weights
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))

# 2. 모델 구조
# model = Sequential()
# model.add(Dense(256, input_dim = 10, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.1))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.1))
# model.add(Dense(4, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.1))
# model.add(Dense(1, activation='sigmoid')) 


input1 = Input(shape=[10,]) # Sequential 모델의 input_shape랑 같음
dense1 = Dense(400, activation='relu')(input1) #ys1 summary에서 이름이 바뀜
Batch1 = BatchNormalization()(dense1)
drop1 = Dropout(0.3)(Batch1)
dense2 = Dense(400,activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(300,activation='relu')(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(300,activation='relu')(drop3)
drop4 = Dropout(0.3)(dense4)
dense5 = Dense(300,activation='relu')(drop4)
drop5= Dropout(0.3)(dense5)
dense6 = Dense(50,activation='relu')(drop5)
drop6 = Dropout(0.3)(dense6)
dense7 = Dense(20,activation='relu')(drop6)
drop7 = Dropout(0.3)(dense7)
output1= Dense(1, activation='sigmoid')(drop7)

model = Model(inputs=input1, outputs=output1)


# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(
    monitor='val_loss', # 평가 지표로 확인하겠다
    mode= 'min', # 최대값 max, 알아서 찾아줘:auto
    patience=20, # 10번까지 초과해도 넘어가겠다
    restore_best_weights=True, # val_loss값이 가장 낮은 값으로 저장 해놓겠다(False시 => 최소값 이후 10번째 값으로 그냥 잡는다.)
)
import datetime
date = datetime.datetime.now()
print(date)
print(type(date))
date = date.strftime('%m%d_%H%M%S')
print(date)
print(type(date)) # <class 'str'>
filename = '{epoch:04d}-{val_loss:.4f}Keras28_MCP_save_08_kaggle_bank.hdf5'
path = '.\_save\Keras28_mcp\\08_kaggle_bank\\'
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only= True,
    filepath=path+filename
)
#model.compile(loss='mse', optimizer='adam')
model.compile(loss='mse', optimizer='adam', metrics=['acc']) # 이진 분류 loss는 무조건
hist = model.fit(x_train, y_train,epochs= 50, batch_size= 12,verbose=2,validation_split=0.1, callbacks=[es,mcp],class_weight=class_weights,)

# 4. 평가 예측
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
print(y_submit)
#y_submit =  (y_submit > 0.5).astype(int)
#y_pred = [1 if y > 0.5 else 0 for y in y_submit]
######## submission.csv 파일 만들기 //count컬럼값만 넣어주기########
submission_csv['Exited'] = y_submit
#print(submission_csv)

#################### csv파일 만들기 #########################
submission_csv.to_csv(path + 'submission_0527_15.csv') # CSV 만들기.

import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

plt.figure(figsize=(9,6)) # 9 x 6
plt.plot(hist.history['loss'], c = 'red', label = 'loss') # x축은 epochs, y값만 넣으면 시간순으로 그림 그림
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss')
plt.plot(hist.history['acc'], c = 'green', label = 'acc')
plt.title('bank Loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right') # 우측 상단에 라벨 표시
plt.grid() # 격자 표시
plt.show()

# [0.12178796529769897, 0.835433840751648]