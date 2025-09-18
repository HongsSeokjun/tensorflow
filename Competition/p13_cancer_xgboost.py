#https://dacon.io/competitions/official/236488/overview/description

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
import pickle
import lightgbm as lgb
# 1. 데이터
path = './_data/dacon/Cancer/'

train_csv = pd.read_csv(path+'train.csv',index_col=0)
test_csv =pd.read_csv(path+'test.csv', index_col=0)
submission_csv = pd.read_csv(path+'sample_submission.csv',index_col=0)


# cancercount = ((train_csv['Cancer'] == 1)).sum()
# count = (
#     (train_csv['Cancer'] == 1) &
#     (
#         (train_csv['Family_Background'] == 'Positive') |
#         (train_csv['Iodine_Deficiency'] == 'Sufficient') |
#         (train_csv['Radiation_History'] == 'Exposed')
#     )
# ).sum()
# non_cancer_mean = train_csv[train_csv['Cancer'] == 0]['Nodule_Size'].mean()
# print(count) #10459
# print(non_cancer_mean) #jpan 0, chn 1380
# # print(train_csv['Country'].value_counts()) # 10개국
# # cancer_df = train_csv[train_csv['Cancer'] == 1]
# # print(cancer_df)
# # 결과 저장
# # cancer_df.to_csv(path+'cancer_only.csv', index=False)  # ← 새 파일 이름
# exit()

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

train_csv['Metabolic_Risk_AND'] = train_csv['Weight_Risk'] & train_csv['Diabetes']
test_csv['Metabolic_Risk_AND'] = test_csv['Weight_Risk'] & test_csv['Diabetes']

x = train_csv.drop(['Cancer','TSH_Result','T4_Result','T3_Result','Weight_Risk','Diabetes','Country',], axis=1)#'Diabetes'
y = train_csv['Cancer']
test_csv = test_csv.drop(['TSH_Result','T4_Result','T3_Result','Weight_Risk','Diabetes','Country',],axis=1)

### 3. 교차 검증 및 SMOTE 적용 (데이터 누수 방지) ###

N_SPLITS = 5 # K-Fold Cross-Validation 분할 수

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=190)
n_classes = len(np.unique(y))
# 각 폴드에서의 모델 예측 확률 및 실제 라벨을 저장할 리스트
oof_preds_proba = np.zeros((x.shape[0],n_classes))
oof_labels = np.zeros(x.shape[0])

# 최종 테스트 세트 예측 확률을 위한 리스트
test_preds_proba_list = []

xgb_models = []
lgb_models = []
cat_models = []

print("\n" + "="*60)
print(f"K-Fold Cross-Validation ({N_SPLITS} Folds)")
print("="*60)

for fold, (train_idx, val_idx) in enumerate(skf.split(x, y)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
    X_train_fold, X_val_fold = x.iloc[train_idx], x.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    # 폴드 내에서 SMOTE 적용
    # original_distribution = Counter(y_train_fold)
    # minority_classes = [class_id for class_id, count in original_distribution.items() if count <= 100]

    # target_distribution = {}
    # for class_id, count in original_distribution.items():
    #     if class_id in minority_classes and count <= 100:
    #         target_distribution[class_id] = min(count * 3, 300) # 최대 300개로 제한
    #     else:
    #         target_distribution[class_id] = count

    # smote = SMOTE(sampling_strategy=target_distribution, random_state=190, k_neighbors=3)
    # X_train_fold_balanced, y_train_fold_balanced = smote.fit_resample(X_train_fold, y_train_fold)

    # print(f"  원본 학습 폴드 분포: {original_distribution}")
    # print(f"  SMOTE 후 학습 폴드 분포: {Counter(y_train_fold_balanced)}")

    # 클래스 가중치 계산 (증강된 데이터 기준)
    classes = np.unique(y_train_fold)
    class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train_fold
    )
    class_weight_dict = dict(zip(classes, class_weights))  # ✅ 이게 정석
    sample_weights = np.array([class_weight_dict[label] for label in y_train_fold])
    # 클래스 가중치 계산 (기존 방식)
    # classes = np.unique(y_train_fold)
    # class_weights = compute_class_weight(
    # class_weight='balanced',
    # classes=classes,
    # y=y_train_fold
    # )
    # class_weight_dict = dict(zip(classes, class_weights))

    # # 기본 sample weight 설정
    # sample_weights = np.array([class_weight_dict[label] for label in y_train_fold])

    # # 특정 조건 만족 시 가중치 추가 (예: '특정칼럼' == 'A' and target==1일 때 2배)
    # condition = ((X_train_fold['특징'] == 0) & (y_train_fold == 1))
    # condition1 = ((X_train_fold['특징1'] == 0) & (y_train_fold == 1))

    # sample_weights[condition] *= 1.1
    # sample_weights[condition1] *= 1.2
    # 모델 학습
    print("  Training models...")

    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.06,
        objective='binary:logistic',
        use_label_encoder=False,
        random_state=190,
        verbosity=0,
        eval_metric='logloss',
        early_stopping_rounds=50,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    xgb_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights,eval_set=[(X_val_fold, y_val_fold)])
    xgb_models.append(xgb_model)

    # LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.06,
        objective='binary',
        metric='binary_logloss',
        random_state=190,
        verbosity=-1,
        force_col_wise=True, # 데이터셋이 작아서 경고 나올 수 있음.
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    lgb_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights,eval_set=[(X_val_fold, y_val_fold)],callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
    lgb_models.append(lgb_model)

    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.06,
        loss_function='Logloss',
        random_seed=190,
        verbose=False,
        l2_leaf_reg=3,
        random_strength=2,
        # 'protocol' 컬럼이 Categorical임을 CatBoost에게 알려줍니다.
        # X_train_processed는 numpy array이므로, feature_names에서 인덱스를 찾아야 합니다.
        # 이 예시에서는 protocol_col_index를 사용합니다.
        #cat_features=[protocol_col_index] if protocol_col_index != -1 else []
    )
    cat_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights,eval_set=[(X_val_fold, y_val_fold)],early_stopping_rounds=50)
    cat_models.append(cat_model)

# 각 fold 모델 저장
    with open(f'xgb_model_fold{fold+1}.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)

    with open(f'lgb_model_fold{fold+1}.pkl', 'wb') as f:
        pickle.dump(lgb_model, f)

    with open(f'cat_model_fold{fold+1}.pkl', 'wb') as f:
        pickle.dump(cat_model, f)
    
    # OOF (Out-Of-Fold) 예측 저장 (앙상블 스태킹에 활용 가능)
    # 각 모델의 예측 확률을 사용하여 평균 소프트 보팅 수행
    oof_xgb_proba = xgb_model.predict_proba(X_val_fold)
    oof_lgb_proba = lgb_model.predict_proba(X_val_fold)
    oof_cat_proba = cat_model.predict_proba(X_val_fold)
    weights = {'xgb': 1.0, 'lgbm': 1.0, 'cat': 1.0}
    total_weight = sum(weights.values())
    oof_avg_proba = (oof_xgb_proba*weights['xgb'] + oof_lgb_proba*weights['lgbm']  + oof_cat_proba*weights['cat'] ) / total_weight
    oof_preds_proba[val_idx] = oof_avg_proba
    oof_labels[val_idx] = y_val_fold
    
    # 최종 테스트 세트에 대한 예측 (각 폴드 모델의 예측을 저장)
    test_preds_proba_list.append(
        (xgb_model.predict_proba(test_csv)*weights['xgb'] +
         lgb_model.predict_proba(test_csv)*weights['lgbm'] +
         cat_model.predict_proba(test_csv)*weights['cat']) / total_weight
    )
# ✅ ensemble_info dict에 필요한 정보 모두 저장
ensemble_info = {
    'xgb_model': xgb_model,
    'lgb_model': lgb_model,
    'cat_model': cat_model,
    'weights': weights,               # 가중치 포함
    'oof_proba': oof_avg_proba,       # fold OOF 확률
    'oof_labels': y_val_fold          # fold OOF 실제값
}

with open(f'ensemble_fold{fold+1}.pkl', 'wb') as f:
    pickle.dump(ensemble_info, f)
    
# OOF 예측 성능 평가
oof_preds_final = np.argmax(oof_preds_proba, axis=1)
oof_accuracy = accuracy_score(oof_labels, oof_preds_final)
oof_f1_macro = f1_score(oof_labels, oof_preds_final, average='macro')
oof_f1_weighted = f1_score(oof_labels, oof_preds_final, average='weighted')

print("\n" + "="*60)
print("OOF (Out-Of-Fold) Ensemble Performance")
print("="*60)
print(f"Accuracy: {oof_accuracy:.4f}")
print(f"Macro F1: {oof_f1_macro:.4f}")
print(f"Weighted F1: {oof_f1_weighted:.4f}")

# K-Fold 앙상블 (Soft Voting)을 통한 최종 예측
# test_preds_proba_list에 저장된 각 폴드 모델의 예측 확률을 평균
final_test_avg_proba = np.mean(test_preds_proba_list, axis=0)
final_test_preds_encoded = np.argmax(final_test_avg_proba, axis=1)

y_submit = final_test_preds_encoded
print(y_submit)

submission_csv['Cancer'] = y_submit

#################### csv파일 만들기 #########################
submission_csv.to_csv(path + 'submission_0622_2.csv') # CSV 만들기.
