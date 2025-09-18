#https://dacon.io/competitions/official/236488/overview/description

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,RobustScaler,OneHotEncoder,StandardScaler
from sklearn.utils import class_weight
import matplotlib.pylab as plt
import seaborn as sns
import xgboost as xgb
#from xgboost import XGBClassifier
# 1. 데이터
path = './_data/dacon/Cancer/'

train_csv = pd.read_csv(path+'train.csv',index_col=0)
test_csv =pd.read_csv(path+'test.csv', index_col=0)
submission_csv = pd.read_csv(path+'sample_submission.csv',index_col=0)

label_cols = ['Gender', 'Country','Race','Family_Background','Smoke',
              'Weight_Risk','Diabetes','Iodine_Deficiency','Radiation_History']
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col] = le.transform(test_csv[col])
    label_encoders[col] = le  # 나중에 inverse_transform 할 때 쓰기 위해 저장
    

x = train_csv.drop(['Cancer','Diabetes'], axis=1)#'Diabetes'
#x = train_csv[['Gender','Country','Smoke','Weight_Risk','Diabetes','Iodine_Deficiency','Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']]
y = train_csv['Cancer']
#test_csv = test_csv[['Gender','Country','Smoke','Weight_Risk','Diabetes','Iodine_Deficiency','Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']]
test_csv = test_csv.drop(['Diabetes'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.01, random_state= 190,stratify=y)

standard = StandardScaler() #표준화
scaler = MinMaxScaler() # 정규화
Robu =RobustScaler()
x_train[['Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']] = scaler.fit_transform(x_train[['Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']])        # train 데이터에 맞춰서 스케일링
x_test[['Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']]= scaler.transform(x_test[['Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']]) # test 데이터는 transform만!
test_csv[['Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']] = scaler.transform(test_csv[['Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']])

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y =y_train
)
class_weights = dict(enumerate(weights))

#2. 모델 구조
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric='logloss',
    use_label_encoder=False, 
    random_state=97
)

# 3. 컴파일 ,훈련
xgb_model.fit(x_train, y_train)#,eval_set=[(x_test, y_test)], early_stopping_rounds=50)
xgb_model.save_model('xgb_model_250604.json')  # 또는 .model 확장자
# 4. 평가 예측

#_predict = model.predict(x_test)
#y_predict =  (y_predict > 0.5).astype(int)
y_predict = xgb_model.predict(x_test)
f1_score1 = f1_score(y_test, y_predict)
print('f1_score :', f1_score1)

y_submit = xgb_model.predict(test_csv)
submission_csv['Cancer'] = y_submit

#################### csv파일 만들기 #########################
submission_csv.to_csv(path + 'submission_0529_1430.csv') # CSV 만들기.
