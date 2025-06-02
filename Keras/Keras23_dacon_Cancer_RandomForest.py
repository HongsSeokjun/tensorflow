from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,RobustScaler
import numpy as np
import pandas as pd

# 1. 데이터
path = './_data/dacon/Cancer/'

train_csv = pd.read_csv(path+'train.csv',index_col=0)
test_csv =pd.read_csv(path+'test.csv', index_col=0)
submission_csv = pd.read_csv(path+'sample_submission.csv',index_col=0)

label_cols = ['Gender', 'Country','Race','Family_Background','Smoke',
              'Weight_Risk','Diabetes','Iodine_Deficiency','Radiation_History']
label_encoders = {}
#reshape(-1, 1)
for col in label_cols:
    le = LabelEncoder()
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col] = le.transform(test_csv[col])
    label_encoders[col] = le  # 나중에 inverse_transform 할 때 쓰기 위해 저장

x = train_csv.drop(['Cancer','Diabetes'], axis=1)#'Diabetes'
y = train_csv['Cancer']
test_csv = test_csv.drop(['Diabetes'], axis=1)


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.01, random_state= 50,stratify=y)

scaler = MinMaxScaler() # 정규화

x_train[['Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']] = scaler.fit_transform(x_train[['Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']])        # train 데이터에 맞춰서 스케일링
x_test[['Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']]= scaler.transform(x_test[['Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']]) # test 데이터는 transform만!
test_csv[['Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']] = scaler.transform(test_csv[['Age','Nodule_Size','TSH_Result','T4_Result','T3_Result']])

# (86287, 13)
# (86287,)

# 클래스 불균형 가중치 계산 (딥러닝에서 class_weight 쓰셨으니 여기도 반영)
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights_dict = dict(zip(classes, class_weights))

# 모델 정의
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight=class_weights_dict,
    max_depth=None,  # 필요시 제한
    n_jobs=-1
)

# 모델 학습
rf_model.fit(x_train, y_train)

# 검증
y_pred = rf_model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

y_submit = rf_model.predict(test_csv)
print(y_submit)
y_submit =  (y_submit > 0.5).astype(int)
submission_csv['Cancer'] = y_submit

#################### csv파일 만들기 #########################
submission_csv.to_csv(path + 'submission_0602_15.csv') # CSV 만들기.
