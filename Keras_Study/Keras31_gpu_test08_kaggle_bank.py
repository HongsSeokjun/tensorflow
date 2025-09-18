# https://www.kaggle.com/competitions/playground-series-s4e1/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.utils import class_weight
#import seaborn as sns
import matplotlib.pylab as plt
import matplotlib
import tensorflow as tf
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
model = Sequential()
model.add(Dense(256, input_dim = 10, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(4, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid')) 


# 3. 컴파일, 훈련

#model.compile(loss='mse', optimizer='adam')
model.compile(loss='mse', optimizer='adam', metrics=['acc']) # 이진 분류 loss는 무조건

start = time.time()
hist = model.fit(x_train, y_train,epochs= 100, batch_size= 32,verbose=2,validation_split=0.1,class_weight=class_weights,)
end = time.time()
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    print('GPU 있다')
else:
    print('GPU 없다')


print("걸린시간 :",end-start)

# GPU 있다
# 걸린시간 : 4705.467570066452
# GPU 없다
# 걸린시간 : 646.099199295044