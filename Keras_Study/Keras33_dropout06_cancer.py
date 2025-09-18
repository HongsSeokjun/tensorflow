#https://dacon.io/competitions/official/236488/overview/description

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization,LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,RobustScaler,OneHotEncoder,StandardScaler,MaxAbsScaler
from sklearn.utils import class_weight
import matplotlib.pylab as plt
import seaborn as sns
import os
from tensorflow.keras.optimizers import Adam
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

def smoke_gender_risk(smoke, gender):
    if smoke == 'Smoker' and gender == 'F':
        return 1  # 여성 흡연자
    else:
        return 0  # 그 외

train_csv['Smoke_Gender_Risk'] = train_csv.apply(lambda row: smoke_gender_risk(row['Smoke'], row['Gender']), axis=1)
test_csv['Smoke_Gender_Risk'] = test_csv.apply(lambda row: smoke_gender_risk(row['Smoke'], row['Gender']), axis=1)

def radiation_region_risk(radiation, country):
    if radiation == 'Sufficient' and country in ['USA']:  # 방사능 사고 지역 예시
        return 2
    elif radiation == 'Sufficient'and country in ['DEU','JPN','CHN']:
        return 1
    else:
        return 0

rate_cancer = train_csv[train_csv['Cancer']==1]['Family_Background'].value_counts(normalize=True)
rate_non_cancer = train_csv[train_csv['Cancer']==0]['Family_Background'].value_counts(normalize=True)

# 1) 고위험 국가 변수 생성 (IND만 1, 나머지 0)
train_csv['High_Risk_Country'] = train_csv['Country'].apply(lambda x: 1 if x == 'IND' else 0)
test_csv['High_Risk_Country'] = test_csv['Country'].apply(lambda x: 1 if x == 'IND' else 0)

train_csv['Radiation_Region_Risk'] = train_csv.apply(lambda row: radiation_region_risk(row['Iodine_Deficiency'], row['Country']), axis=1)
test_csv['Radiation_Region_Risk'] = test_csv.apply(lambda row: radiation_region_risk(row['Iodine_Deficiency'], row['Country']), axis=1)

label_cols = ['Gender','Race','Family_Background','Smoke',
              'Weight_Risk','Diabetes','Iodine_Deficiency','Radiation_History']
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col] = le.transform(test_csv[col])
    label_encoders[col] = le  # 나중에 inverse_transform 할 때 쓰기 위해 저장


standard = StandardScaler() #표준화
scaler = MinMaxScaler() # 정규화

train_csv[['Age','Nodule_Size']] = scaler.fit_transform(train_csv[['Age','Nodule_Size']])        # train 데이터에 맞춰서 스케일링
test_csv[['Age','Nodule_Size']] = scaler.transform(test_csv[['Age','Nodule_Size']])
train_csv[['T4_Result','T3_Result','TSH_Result']] = standard.fit_transform(train_csv[['T4_Result','T3_Result','TSH_Result']])        # train 데이터에 맞춰서 스케일링
test_csv[['T4_Result','T3_Result','TSH_Result']] = standard.transform(test_csv[['T4_Result','T3_Result','TSH_Result']])

train_csv['Thyroid_Function_Score'] = train_csv['TSH_Result'] + train_csv['T4_Result'] + train_csv['T3_Result']
test_csv['Thyroid_Function_Score'] = test_csv['TSH_Result'] + test_csv['T4_Result'] + test_csv['T3_Result']

train_csv['Metabolic_Risk'] = train_csv['Weight_Risk'] | train_csv['Diabetes']
test_csv['Metabolic_Risk'] = test_csv['Weight_Risk'] | test_csv['Diabetes']


x = train_csv.drop(['Cancer','TSH_Result','T4_Result','T3_Result','Weight_Risk','Diabetes','Country',], axis=1)#'Diabetes'
y = train_csv['Cancer']
test_csv = test_csv.drop(['TSH_Result','T4_Result','T3_Result','Weight_Risk','Diabetes','Country',],axis=1)

# corr = train_csv.corr()  # 변수들 간 상관관계 계산
# plt.figure(figsize=(10,8))
# # sns.boxplot(x=train_csv['Age'])
# sns.heatmap(corr, annot=True, cmap='coolwarm')  # annot=True는 숫자 표시
# plt.show()
# exit()
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=190)

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y =y_train
)
class_weights = dict(enumerate(weights))

#2. 모델 구조
model = Sequential()
model.add(Dense(128, input_dim=15,))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1)) # 또는 ReLU
model.add(Dropout(0.3))
model.add(Dense(64))#activation=LeakyReLU(alpha=0.1)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1)) # 또는 ReLU
model.add(Dropout(0.3))
model.add(Dense(64,))#activation=LeakyReLU(alpha=0.1)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1)) # 또는 ReLU
model.add(Dropout(0.2))
model.add(Dense(32,))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1)) # 또는 ReLU
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일 ,훈련
es = EarlyStopping(
    monitor='val_loss', # 지표를 acc로 잡으면 max로 잡아할때도 있다. => auto로 잡으면 알아서 잡아줌
    mode='min',
    patience= 150,
    restore_best_weights=True,
)

filename = 'cancer0621.hdf5'
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only= True,
    filepath=path+filename
)
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=20,
    verbose=1,
    min_lr=1e-6
)
# optimizers = Adam(learning_rate=0.0008)# RMSprop SGD with momentum
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])#categorical_crossentropy 
hist = model.fit(x_train, y_train, epochs = 1000, batch_size=32, verbose=2, validation_split=0.08, callbacks=[es,mcp, rlr],class_weight=class_weights,)

os.makedirs(path, exist_ok=True)
model_path = os.path.join(path, 'cancer_model.hdf5')
weights_path = os.path.join(path, 'cancer_weights.hdf5')
# 가중치 저장
model.save_weights(weights_path)
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
submission_csv.to_csv(path + 'submission_0622_09.csv') # CSV 만들기.

# import matplotlib.pylab as plt
# import matplotlib
# matplotlib.rcParams['font.family'] = 'Malgun Gothic'

# plt.figure(figsize=(9,6)) # 9 x 6
# plt.plot(hist.history['loss'], c = 'red', label = 'loss') # x축은 epochs, y값만 넣으면 시간순으로 그림 그림
# plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss')
# plt.plot(hist.history['acc'], c = 'green', label = 'acc')
# plt.title('bank Loss')
# plt.xlabel('epoch')
# plt.ylabel('Loss')
# plt.legend(loc='upper right') # 우측 상단에 라벨 표시
# plt.grid() # 격자 표시
# plt.show()

# [0.5192498564720154, 0.896789014339447]
# f1_score : 0.5544554455445544