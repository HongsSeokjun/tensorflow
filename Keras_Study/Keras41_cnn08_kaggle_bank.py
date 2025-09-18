from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization,Conv2D,Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score#불균형 데이터일때
from sklearn.preprocessing import StandardScaler,LabelEncoder
import tensorflow as tf
import matplotlib.pylab as plt
from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint
#1. 데이터
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
print(x.shape)#(165034, 10)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state= 47)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 10, 1, 1)  # (batch, height, width, channel)
x_test = x_test.reshape(-1, 10, 1, 1)  # (batch, height, width, channel)


#2. 모델구성
model = Sequential()
model.add(Conv2D(32,(1,1),input_shape=(10,1,1), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(8,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(
    monitor='val_loss', # 평가 지표로 확인하겠다
    mode= 'min', # 최대값 max, 알아서 찾아줘:auto
    patience=10, # 10번까지 초과해도 넘어가겠다
    restore_best_weights= True # val_loss값이 가장 낮은 값으로 저장 해놓겠다(False시 => 최소값 이후 10번째 값으로 그냥 잡는다.)
)
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train,epochs= 20, batch_size= 4,verbose=2,validation_split=0.1,callbacks=[es])


# 4. 평가 예측

loss = model.evaluate(x_test,y_test)

# r2 = r2_score(y_test, result)
print('loss :',loss)
y_predict = model.predict(x_test)
y_predict =  (y_predict > 0.5).astype(int)
from sklearn.metrics import accuracy_score # 이진만 받을 수 있다
accuracy_score = accuracy_score(y_test, y_predict)
print('acc',accuracy_score)
#r2 : 0.593319922402797