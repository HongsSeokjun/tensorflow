#https://dacon.io/competitions/official/236488/overview/description

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score,precision_recall_curve
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,RobustScaler,StandardScaler
from sklearn.utils import class_weight
import matplotlib.pylab as plt
import seaborn as sns
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
# 1. 데이터
path = './_data/dacon/Cancer/'

train_csv = pd.read_csv(path+'train.csv',index_col=0)
test_csv =pd.read_csv(path+'test.csv', index_col=0)
submission_csv = pd.read_csv(path+'sample_submission.csv',index_col=0)

#print(train_csv.shape) #(87159, 15)
#print(test_csv.shape) #(46204, 14)
# print(train_csv.columns)

label_cols = ['Gender', 'Country','Race','Family_Background','Smoke',
              'Weight_Risk','Diabetes','Iodine_Deficiency','Radiation_History']
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col] = le.transform(test_csv[col])
    label_encoders[col] = le  # 나중에 inverse_transform 할 때 쓰기 위해 저장
    
# Index(['Age', 'Gender', 'Country', 'Race', 'Family_Background',
#        'Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Weight_Risk',
#        'Diabetes', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result',
#        'Cancer'],

# corr = train_csv.corr()  # 변수들 간 상관관계 계산
# plt.figure(figsize=(10,8))
# sns.heatmap(corr, annot=True, cmap='coolwarm')  # annot=True는 숫자 표시
# plt.show()
# exit()
x = train_csv.drop(['Cancer','Age','TSH_Result','T4_Result','Nodule_Size'], axis=1)
y = train_csv['Cancer']
test_csv = test_csv.drop(['Age','TSH_Result','T4_Result','Nodule_Size'], axis=1)
# 4. 중요도 시각화 전용 모델 학습
# ==============================================
standard = StandardScaler() #표준화
scaler = MinMaxScaler() # 정규화
Robu =RobustScaler()

x[['T3_Result']] = scaler.fit_transform(x[['T3_Result']])        # train 데이터에 맞춰서 스케일링
test_csv[['T3_Result']] = scaler.transform(test_csv[['T3_Result']])
# 전처리 완료된 전체 X, y 사용
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=190)
smote = SMOTE(random_state=190)
X_train_res, y_train_res = smote.fit_resample(x_train, y_train)


# x_train[['Age','TSH_Result','T4_Result']] = standard.fit_transform(x_train[['Age','TSH_Result','T4_Result']])        # train 데이터에 맞춰서 스케일링
# x_test[['Age','TSH_Result','T4_Result']]= standard.transform(x_test[['Age','TSH_Result','T4_Result']]) # test 데이터는 transform만!
# test_csv[['Age','TSH_Result','T4_Result']] = standard.transform(test_csv[['Age','TSH_Result','T4_Result']])


model_full = XGBClassifier(random_state=47, eval_metric='logloss')
model_full.fit(X_train_res, y_train_res)
_= model_full
# 5. Feature Importance 시각화 (전체 feature 기준)

importances = model_full.feature_importances_
features = X_train_res.columns.tolist()

feature_importance = list(zip(features, importances))
feature_importance.sort(key=lambda x: x[1], reverse=True)

sorted_features = [f[0] for f in feature_importance]
sorted_importances = [f[1] for f in feature_importance]

# plt.figure(figsize=(10, 6))
# plt.title("Feature Importance (XGBoost - Full Feature Set)")
# plt.barh(range(len(sorted_features)), sorted_importances[::-1], align="center")
# plt.yticks(range(len(sorted_features)), sorted_features[::-1])
# plt.xlabel("Importance Score")
# plt.tight_layout()
# plt.show()
# exit()
# 6. 중요도 낮은 feature 제거 후 재정의
# drop_features = ["T3_Result", "Nodule_Size", "Age", "T4_Result", "TSH_Result"]
# X = X.drop(columns=drop_features)
# X_test_dropped = X_test.drop(columns=drop_features)  # drop된 X_test는 따로 저장
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_res),
    y =y_train_res
)
class_weight_dict = dict(enumerate(weights))
sample_weights = pd.Series(y_train_res).map(class_weight_dict).values
n_classes = len(np.unique(y_train_res))
# 7. 모델 학습 (앙상블: XGB + LGBM + CatBoost)
# ==============================================

# XGBoost
xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=n_classes,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=190,
        verbosity=0
)
# LightGBM
lgbm = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective='multiclass',
        num_class=n_classes,
        metric='multi_logloss',
        random_state=190,
        verbosity=-1,
        force_col_wise=True # 데이터셋이 작아서 경고 나올 수 있음.
)
 # CatBoost
cat = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        loss_function='MultiClass',
        random_seed=190,
        verbose=False,
)

xgb.fit(X_train_res, y_train_res)
lgbm.fit(X_train_res, y_train_res)
cat.fit(X_train_res, y_train_res)

# 8. Threshold 최적화
# ==============================================
xgb_val = xgb.predict_proba(x_test)[:, 1]
lgbm_val = lgbm.predict_proba(x_test)[:, 1]
cat_val = cat.predict_proba(x_test)[:, 1]
weights = {'xgb': 1.0, 'lgbm': 1.0, 'cat': 1.5}
total_weight = sum(weights.values())

ensemble_val = (xgb_val * 1.0 +lgbm_val * 1.0 +cat_val * 1.5) / total_weight

precisions, recalls, thresholds = precision_recall_curve(y_test, ensemble_val)
f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1s)
best_threshold = thresholds[best_idx]

# 4. 평가 예측
xgb_test = xgb.predict_proba(test_csv)[:, 1]
lgbm_test = lgbm.predict_proba(test_csv)[:, 1]
cat_test = cat.predict_proba(test_csv)[:, 1]
ensemble_test = (
    xgb_test * 1.0 +
    lgbm_test * 1.0 +
    cat_test * 1.5
) / total_weight

y_submit = (ensemble_test >= best_threshold).astype(int)
submission_csv['Cancer'] = y_submit

#################### csv파일 만들기 #########################
submission_csv.to_csv(path + 'submission_0610_1430.csv') # CSV 만들기.
