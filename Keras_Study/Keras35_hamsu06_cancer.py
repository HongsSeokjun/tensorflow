#https://dacon.io/competitions/official/236488/overview/description

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization,LeakyReLU,Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,RobustScaler,OneHotEncoder,StandardScaler,MaxAbsScaler
from sklearn.utils import class_weight
import matplotlib.pylab as plt
#import seaborn as sns
# import xgboost as xgb
#from imblearn.over_sampling import SMOTE
#from xgboost import XGBClassifier
# 1. 데이터
path = './_data/dacon/Cancer/'

train_csv = pd.read_csv(path+'train.csv',index_col=0)
test_csv =pd.read_csv(path+'test.csv', index_col=0)
submission_csv = pd.read_csv(path+'sample_submission.csv',index_col=0)

#print(train_csv.shape) #(87159, 15)
#print(test_csv.shape) #(46204, 14)
# print(train_csv.columns)
# Index(['Age', 'Gender', 'Country', 'Race', 'Family_Background',
#        'Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Weight_Risk',
#        'Diabetes', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result',
#        'Cancer'],

label_cols = ['Gender', 'Country','Race','Family_Background','Smoke',
              'Weight_Risk','Diabetes','Iodine_Deficiency','Radiation_History']
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col] = le.transform(test_csv[col])
    label_encoders[col] = le  # 나중에 inverse_transform 할 때 쓰기 위해 저장
    
# train_csv['Race'] = label_encoders['Race'].inverse_transform(train_csv['Race'])
# print(train_csv['Race'])
# exit()
#from sklearn.preprocessing import OneHotEncoder
# train_csv = pd.get_dummies(train_csv, columns=['Country'])
# test_csv = pd.get_dummies(test_csv, columns=['Country'])
# 나이 구간화 함수: 예시 기준
# def age_group(age):
#     if age < 35:
#         return 'young'     # 청년
#     elif age < 60:
#         return 'middle'    # 중년
#     else:
#         return 'old'       # 노년

# # 적용
# train_csv['Age_Group'] = train_csv['Age'].apply(age_group)
# # 원-핫 인코딩 (또는 LabelEncoding도 가능)
# train_csv = pd.get_dummies(train_csv, columns=['Age_Group'], drop_first=True)
# exit()

x = train_csv.drop(['Cancer','Diabetes'], axis=1)#'Diabetes'
#x = train_csv[['Gender','Country','Smoke','Weight_Risk','Diabetes','Iodine_Deficiency','Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']]
y = train_csv['Cancer']
#test_csv = test_csv[['Gender','Country','Smoke','Weight_Risk','Diabetes','Iodine_Deficiency','Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']]
test_csv = test_csv.drop(['Diabetes'], axis=1)
# x = pd.get_dummies(x, columns=['Country', 'Race'], drop_first=True)
# test_csv = pd.get_dummies(test_csv, columns=['Country', 'Race'], drop_first=True)

# def clip_outliers(df, column):
#     Q1 = df[column].quantile(0.25)
#     Q3 = df[column].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
    
#     print(f'{column} 이상치 하한: {lower_bound}, 상한: {upper_bound}')
    
#     # 클리핑 적용 (이상치는 하한 또는 상한 값으로 조정)
#     df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
#     return df

# # 예시로 Nodule_Size와 TSH_Result 처리
# x = clip_outliers(x, 'Nodule_Size')
# x = clip_outliers(x, 'TSH_Result')

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.01, random_state= 190,stratify=y)

standard = StandardScaler() #표준화
scaler = MinMaxScaler() # 정규화
Robu =RobustScaler()
MaxAbsscaler = MaxAbsScaler()
x_train[['Nodule_Size','T3_Result']] = scaler.fit_transform(x_train[['Nodule_Size','T3_Result']])        # train 데이터에 맞춰서 스케일링
x_test[['Nodule_Size','T3_Result']]= scaler.transform(x_test[['Nodule_Size','T3_Result']]) # test 데이터는 transform만!
test_csv[['Nodule_Size','T3_Result']] = scaler.transform(test_csv[['Nodule_Size','T3_Result']])

x_train[['Age','TSH_Result','T4_Result']] = standard.fit_transform(x_train[['Age','TSH_Result','T4_Result']])        # train 데이터에 맞춰서 스케일링
x_test[['Age','TSH_Result','T4_Result']]= standard.transform(x_test[['Age','TSH_Result','T4_Result']]) # test 데이터는 transform만!
test_csv[['Age','TSH_Result','T4_Result']] = standard.transform(test_csv[['Age','TSH_Result','T4_Result']])

#smote = SMOTE(random_state=47)
# x_train, y_train = smote.fit_resample(x_train, y_train)

# x_train = scaler.fit_transform(x_train)        # train 데이터에 맞춰서 스케일링
# x_test= scaler.transform(x_test) # test 데이터는 transform만!
# test_csv = scaler.transform(test_csv)

# corr = train_csv.corr()  # 변수들 간 상관관계 계산
# plt.figure(figsize=(10,8))
# #sns.boxplot(x=train_csv['Age'])
# sns.heatmap(corr, annot=True, cmap='coolwarm')  # annot=True는 숫자 표시
# plt.show()
# exit()
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y =y_train
)
class_weights = dict(enumerate(weights))

#2. 모델 구조
# model = Sequential()
# model.add(Dense(128, input_dim=13, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))
# model.add(Dense(64,   activation='relu'))#activation=LeakyReLU(alpha=0.1)))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))
# model.add(Dense(32,   activation='relu'))#activation=LeakyReLU(alpha=0.1)))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(16,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))

# 2-2. 모델구성 (함수형)
input1 = Input(shape=[13,]) # Sequential 모델의 input_shape랑 같음
dense1 = Dense(128, activation='relu')(input1) #ys1 summary에서 이름이 바뀜
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(64,activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(32,activation='relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(16,activation='relu')(drop3)
drop4 = Dropout(0.2)(dense4)
output1= Dense(1, activation='sigmoid')(drop4)

model = Model(inputs=input1, outputs=output1)


# 3. 컴파일 ,훈련
es = EarlyStopping(
    monitor='val_loss', # 지표를 acc로 잡으면 max로 잡아할때도 있다. => auto로 잡으면 알아서 잡아줌
    mode='min',
    patience= 100,
    restore_best_weights=True,
)

filename = 'cancer.hdf5'
path = '.\_save\Keras28_mcp\\06_cancer\\'
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only= True,
    filepath=path+filename
)

#optimizers = Adam(learning_rate=0.0005)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])#categorical_crossentropy 
hist = model.fit(x_train, y_train, epochs = 10, batch_size=16, verbose=2, validation_split=0.08, callbacks=[es,mcp],class_weight=class_weights,)

# 4. 평가 예측
results = model.evaluate(x_test, y_test)
print(results)

#_predict = model.predict(x_test)
#y_predict =  (y_predict > 0.5).astype(int)
y_predict = model.predict(x_test)
y_predict =  (y_predict > 0.5).astype(int)
f1_score1 = f1_score(y_test, y_predict)
print('f1_score :', f1_score1)

y_submit = model.predict(test_csv)
print(y_submit)
y_submit =  (y_submit > 0.5).astype(int)
submission_csv['Cancer'] = y_submit

#################### csv파일 만들기 #########################
submission_csv.to_csv(path + 'submission_0529_1430.csv') # CSV 만들기.

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

# [0.5192498564720154, 0.896789014339447]
# f1_score : 0.5544554455445544