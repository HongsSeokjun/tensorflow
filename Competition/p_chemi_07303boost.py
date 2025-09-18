#Boost up AI 2025:​ 신약개발경진대회
import os
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, DataStructs
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import optuna
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
import multiprocessing
from sklearn.utils.class_weight import compute_class_weight
import joblib
import json
import os
from lightgbm import early_stopping
warnings.filterwarnings('ignore')

SEED = 222
np.random.seed(SEED)
torch.manual_seed(SEED)

path = 'C:\study25jun\_data\dacon\\Chemical\\open\\'

# --- Morgan Fingerprint (512 bits로 변경) ---
def get_morgan_fp(mol, radius=2, n_bits=512):
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# --- 화학적 피처 추출 ---
def get_physchem_features(mol):
    return [
        Descriptors.MolWt(mol),
        Descriptors.ExactMolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.HeavyAtomMolWt(mol),
        Descriptors.MolMR(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NOCount(mol),
        Descriptors.NHOHCount(mol),
        Descriptors.RingCount(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumSaturatedRings(mol),
        Descriptors.NumValenceElectrons(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumRadicalElectrons(mol),
        Descriptors.NumAromaticHeterocycles(mol),
        Descriptors.NumAliphaticHeterocycles(mol),
        Descriptors.NumSaturatedHeterocycles(mol),
        Descriptors.TPSA(mol),
        rdMolDescriptors.CalcLabuteASA(mol),
        Descriptors.FpDensityMorgan1(mol),
        Descriptors.FpDensityMorgan2(mol),
        Descriptors.FpDensityMorgan3(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        Descriptors.BertzCT(mol),
        Descriptors.HallKierAlpha(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumHeteroatoms(mol),
        Descriptors.MaxPartialCharge(mol),
        Descriptors.MinPartialCharge(mol),
        Descriptors.MaxAbsPartialCharge(mol),
        Descriptors.MinAbsPartialCharge(mol),
        Descriptors.Ipc(mol),
    ]

# --- SMILES to Feature ---
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        physchem = get_physchem_features(mol)
        morgan = get_morgan_fp(mol)
        return physchem + list(morgan)
    except:
        return None

# --- 병렬 처리 ---
def extract_features_parallel(df, num_workers=4):
    pool = multiprocessing.Pool(processes=num_workers)
    features = pool.map(smiles_to_features, df['Canonical_Smiles'])
    pool.close()
    pool.join()
    return features

def prepare_features(df, is_train=True, num_workers=4):
    features = extract_features_parallel(df, num_workers)
    filtered_features = []
    targets = []
    indices = []
    for idx, feat in enumerate(features):
        if feat is None:
            continue
        filtered_features.append(feat)
        indices.append(idx)
        if is_train:
            targets.append(df.iloc[idx]['Inhibition'])
    filtered_features = np.array(filtered_features)
    if is_train:
        return filtered_features, np.array(targets), indices
    return filtered_features, indices

# --- 커스텀 평가 지표 함수 ---
def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    range_y = np.max(y_true) - np.min(y_true)
    normalized_rmse = rmse / range_y if range_y != 0 else 0
    normalized_rmse = min(normalized_rmse, 1)
    corr, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (0, 0)
    score = 0.5 * (1 - normalized_rmse) + 0.5 * corr
    return normalized_rmse, corr, score

# --- XGBoost Optuna 목적함수 ---
def xgb_objective(trial, X, y):
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist',  # GPU 사용 옵션
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1e-1, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1e-1, log=True),
        'random_state': SEED,
        'verbosity': 0,
        
    }
    model = xgb.XGBRegressor(**params)
    kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = []
    for tr_idx, val_idx in kf.split(X):
        model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])],early_stopping_rounds=80)
        pred = model.predict(X[val_idx])
        _, _, sc = compute_metrics(y[val_idx], pred)
        scores.append(sc)
    return -np.mean(scores)

# --- LightGBM Optuna 목적함수 ---
def lgb_objective(trial, X, y):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'device': 'gpu',           # GPU 사용 옵션
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 1e-1, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 1e-1, log=True),
        'seed': SEED
    }
    model = lgb.LGBMRegressor(**params)
    kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = []
    for tr_idx, val_idx in kf.split(X):
        model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])],callbacks=[early_stopping(stopping_rounds=80)])
        pred = model.predict(X[val_idx])
        _, _, sc = compute_metrics(y[val_idx], pred)
        scores.append(sc)
    return -np.mean(scores)

# --- CatBoost Optuna 목적함수 ---
def cat_objective(trial, X, y):
    params = {
        'iterations': trial.suggest_int('iterations', 200, 600),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 10, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'verbose': False,
        'random_seed': SEED,
        'task_type': 'GPU'   # GPU 사용 옵션
    }
    model = cb.CatBoostRegressor(**params)
    kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = []
    for tr_idx, val_idx in kf.split(X):
        model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])],early_stopping_rounds=80)
        pred = model.predict(X[val_idx])
        _, _, sc = compute_metrics(y[val_idx], pred)
        scores.append(sc)
    return -np.mean(scores)
# 저장 함수 정의
def save_study_info(study, filename_prefix):
    import os
    best = study.best_trial
    info = {
        'best_params': best.params,
        'best_value': best.value,
        'best_trial_number': best.number
    }

    # 저장할 전체 경로 만들기
    os.makedirs(path, exist_ok=True)  # path는 외부에서 정의되어 있다고 가정
    full_path = os.path.join(path, f'{filename_prefix}_optuna_best.json')

    with open(full_path, 'w') as f:
        json.dump(info, f, indent=4)

    print(f"[✔] {filename_prefix} best trial 저장 완료: {full_path}")
    
def main():
    train = pd.read_csv(path + 'filtered_train.csv', index_col=0)
    test = pd.read_csv(path + 'test.csv', index_col=0)
    submission = pd.read_csv(path + 'sample_submission.csv')

    train = train.dropna().reset_index(drop=True)
    test = test.fillna(test.median(numeric_only=True))

    print("피처 추출 중... CPU 병렬 처리로 진행")
    X_train, y_train, train_idx = prepare_features(train, is_train=True, num_workers=4)
    X_test, test_idx = prepare_features(test, is_train=False, num_workers=4)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)



    print("[Optuna] XGBoost 튜닝 시작...")
    study_xgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study_xgb.optimize(lambda trial: xgb_objective(trial, X_train, y_train), n_trials=200)
    print("[Optuna] XGBoost 최적 파라미터:", study_xgb.best_params)
    save_study_info(study_xgb, 'xgboost')

    print("[Optuna] LightGBM 튜닝 시작...")
    study_lgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study_lgb.optimize(lambda trial: lgb_objective(trial, X_train, y_train), n_trials=200)
    print("[Optuna] LightGBM 최적 파라미터:", study_lgb.best_params)
    save_study_info(study_lgb, 'lightgbm')

    print("[Optuna] CatBoost 튜닝 시작...")
    study_cat = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study_cat.optimize(lambda trial: cat_objective(trial, X_train, y_train), n_trials=200)
    print("[Optuna] CatBoost 최적 파라미터:", study_cat.best_params)
    save_study_info(study_cat, 'catboost')

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    # 각 모델별 fold별 예측 저장
    models_xgb, models_lgb, models_cat = [], [], []
    preds_xgb_folds, preds_lgb_folds, preds_cat_folds = [], [], []

    val_scores_xgb, val_scores_lgb, val_scores_cat = [], [], []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
            
        
        # ----- XGBoost -----
        model_xgb = xgb.XGBRegressor(
            **study_xgb.best_params,
            random_state=SEED + fold,
            verbosity=0,
            tree_method='gpu_hist'
        )
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False,early_stopping_rounds=80,)
        models_xgb.append(model_xgb)
        preds_xgb_folds.append(model_xgb.predict(X_test))

        val_pred = model_xgb.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - val_pred)**2))
        val_scores_xgb.append(rmse)
        print(f"📉 XGBoost Fold {fold} RMSE: {rmse:.4f}")

        # ----- LightGBM -----
        model_lgb = lgb.LGBMRegressor(
            **study_lgb.best_params,
            random_state=SEED + fold,
            device='gpu'
        )
        model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],callbacks=[early_stopping(stopping_rounds=80)])
        models_lgb.append(model_lgb)
        preds_lgb_folds.append(model_lgb.predict(X_test))

        val_pred = model_lgb.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - val_pred)**2))
        val_scores_lgb.append(rmse)
        print(f"📉 LightGBM Fold {fold} RMSE: {rmse:.4f}")

        # ----- CatBoost -----
        model_cat = cb.CatBoostRegressor(
            **study_cat.best_params,
            random_seed=SEED + fold,
            verbose=False,
            task_type='GPU'
        )
        model_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False,early_stopping_rounds=80,)
        models_cat.append(model_cat)
        preds_cat_folds.append(model_cat.predict(X_test))

        val_pred = model_cat.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - val_pred)**2))
        val_scores_cat.append(rmse)
        print(f"📉 CatBoost Fold {fold} RMSE: {rmse:.4f}")

    # ----- Fold 평균 예측 -----
    preds_xgb = np.mean(preds_xgb_folds, axis=0)
    preds_lgb = np.mean(preds_lgb_folds, axis=0)
    preds_cat = np.mean(preds_cat_folds, axis=0)

    # ----- 가중치 계산 -----
    score_xgb = -study_xgb.best_value
    score_lgb = -study_lgb.best_value
    score_cat = -study_cat.best_value

    scores = [score_xgb, score_lgb, score_cat]
    total = sum(scores)
    if total == 0:
        w_xgb = w_lgb = w_cat = 1/3
    else:
        w_xgb = score_xgb / total
        w_lgb = score_lgb / total
        w_cat = score_cat / total

    print(f"\n🎯 앙상블 가중치 - XGB: {w_xgb:.3f}, LGB: {w_lgb:.3f}, CAT: {w_cat:.3f}")

    # ----- 최종 앙상블 예측 -----
    preds_ensemble = preds_xgb * w_xgb + preds_lgb * w_lgb + preds_cat * w_cat
    # 음수 제거 (0으로 클리핑)
    preds_ensemble = np.clip(preds_ensemble, 0.0, None)
    # 제출 파일 저장
    submission.loc[test_idx, 'Inhibition'] = preds_ensemble
    submission.to_csv(path + 'submission_final_3boost_kfold.csv', index=False)
    print("✅ 제출 파일 저장 완료: submission_final_3boost_kfold.csv")

    # ----- 모델 저장 -----
    os.makedirs(path, exist_ok=True)
    print("💾 모델 및 가중치 저장 중...")

    for i, model in enumerate(models_xgb):
        joblib.dump(model, f"{path}model_xgb_fold{i}.pkl")
    for i, model in enumerate(models_lgb):
        joblib.dump(model, f"{path}model_lgb_fold{i}.pkl")
    for i, model in enumerate(models_cat):
        joblib.dump(model, f"{path}model_cat_fold{i}.pkl")

    weights = {'xgb': w_xgb, 'lgb': w_lgb, 'cat': w_cat}
    with open(path + 'ensemble_weights.json', 'w') as f:
        json.dump(weights, f, indent=4)

    print("✅ 모든 모델과 가중치 저장 완료.")
    # --- 최종 평가 (학습 데이터 기준) ---
    # 모든 모델은 전체 X_train에서 학습했기 때문에 y_train과 예측값 비교 가능
    preds_train_xgb = model_xgb.predict(X_train)
    preds_train_lgb = model_lgb.predict(X_train)
    preds_train_cat = model_cat.predict(X_train)
    preds_train_ensemble = preds_train_xgb * w_xgb + preds_train_lgb * w_lgb + preds_train_cat * w_cat
    preds_train_ensemble = np.clip(preds_train_ensemble, 0.0, None)

    # 평가 지표 출력
    normalized_rmse, corr, final_score = compute_metrics(y_train, preds_train_ensemble)
    print(f"\n[최종 학습 데이터 평가]")
    print(f"✅ Normalized RMSE: {normalized_rmse:.4f}")
    print(f"✅ Pearson Correlation: {corr:.4f}")
    print(f"✅ Combined Score: {final_score:.4f}\n")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

# [최종 학습 데이터 평가]
# ✅ Normalized RMSE: 0.1495
# ✅ Pearson Correlation: 0.8328
# ✅ Combined Score: 0.8416

# [최종 학습 데이터 평가]
# ✅ Normalized RMSE: 0.1458
# ✅ Pearson Correlation: 0.8436

# ✅ Combined Score: 0.8489
