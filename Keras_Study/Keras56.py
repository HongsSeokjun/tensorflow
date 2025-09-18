# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, BatchNormalization,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import LabelEncoder,RobustScaler,StandardScaler,MaxAbsScaler
from sklearn.utils import class_weight

path = 'C:\study25jun\_data\kaggle\jena_clime\\'
train_csv = pd.read_csv(path+'jena_climate_2009_2016.csv') #"T (degC)"
#2016.12.31 'O' 144개 찾기 
# 정리해보면 31.12.2015 144개를 출력으로 해야하고 그 위의 데이터만 사용할 수 있다.
# 위의 데이터를 이용해서 144개를 예측하면 되는거야
# 최종 y Data Time , wd (deg)

# Data Time , wd (deg)
# print(train_csv.columns) 
# Index(['Date Time', 'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)',       
#        'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
#        'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
#        'wd (deg)'],
#       dtype='object')

# print(train_csv.shape) #(420551, 15) 'wd_sin', 'wd_cos'비제거도 테스트
timesteps = 24
y_test_1 = train_csv[ (train_csv['Date Time'].str.contains("31.12.2016", regex=False)) &
    (train_csv['Date Time'] != "31.12.2016 00:00:00")]['wd (deg)'].copy()
y_test_2 = train_csv[train_csv['Date Time'].str.contains("01.01.2017")]['wd (deg)'].copy()

train_csv1 = train_csv

train_csv['wd_sin'] = np.sin(np.radians(train_csv['wd (deg)']))
train_csv['wd_cos'] = np.cos(np.radians(train_csv['wd (deg)']))

train_csv = train_csv[
    ~train_csv['Date Time'].str.startswith("31.12.2016") | 
    (train_csv['Date Time'] == "31.12.2016 00:00:00")# 30일 하루치하고 31 0시
]
train_csv = train_csv[~train_csv['Date Time'].str.contains("01.01.2017")]

y_test = pd.concat([y_test_1, y_test_2], ignore_index=True) #ignore_index 인덱스를 새로 매겨줘

train_csv['Date Time'] = pd.to_datetime(train_csv['Date Time'])
train_csv['hour'] = train_csv['Date Time'].dt.hour
train_csv['month'] = train_csv['Date Time'].dt.month
train_csv['day'] = train_csv['Date Time'].dt.day
train_csv['minute'] = train_csv['Date Time'].dt.minute
train_csv['weekday'] = train_csv['Date Time'].dt.weekday  # 월=0, 일=6

train_csv1['Date Time'] = pd.to_datetime(train_csv1['Date Time'])
train_csv1['hour'] = train_csv1['Date Time'].dt.hour
train_csv1['month'] = train_csv1['Date Time'].dt.month
train_csv1['day'] = train_csv1['Date Time'].dt.day
train_csv1['minute'] = train_csv1['Date Time'].dt.minute
train_csv1['weekday'] = train_csv1['Date Time'].dt.weekday  # 월=0, 일=6

cols = ['T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)', 
        'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 
        'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',]# 'wd (deg)']

Stan_encoders = {}

for col in cols:
    standard = StandardScaler() #표준화
    train_csv[col] = standard.fit_transform(train_csv[[col]])
    train_csv1[col] = standard.fit_transform(train_csv1[[col]])
    Stan_encoders[col] = standard  # 나중에 inverse_transform 할 때 쓰기 위해 저장
    
x = train_csv.drop(['Date Time','wd (deg)','wd_sin', 'wd_cos'], axis=1).values  # 칼럼제거버전
# x = train_csv.drop(['Date Time'], axis=1).values
y = train_csv[['wd_sin', 'wd_cos']].values               # numpy
# print(x)
# print(x.shape) #(24, 15)
# print(y)
# print(y.shape)
# exit()

y = y[timesteps-1:]      

# x_test = train_csv['Date Time'] 

def split_x_stride(x, window_size, stride):
    sequences = []
    for start in range(0, len(x) - window_size + 1, stride):
        subset = x[start : start + window_size]
        sequences.append(subset)
    return np.array(sequences)

target_horizon =144
x = x[: - (timesteps + (target_horizon - 1))]


x = split_x_stride(x, timesteps, 3)
y = y[1 : len(x)*3 + 1 : 3] # y[start : stop : step]
# start: 슬라이싱을 시작할 인덱스

# stop: 슬라이싱을 멈출 인덱스 (해당 인덱스는 포함되지 않음)

# step: 슬라이싱 간격 (몇 칸씩 띄우는지)

print(x.shape)#(140073, 24, 18)
print(y)
print(y.shape) #(140073, 2)
exit()
# x_test를 만들기 위해 필요한 구간 (24 timesteps 이전부터 예측 대상인 144포인트까지)

# 최종 결과를 얻어야하는 데이터
total_steps = 144 + timesteps -1
x_test_source = train_csv1[-total_steps:]
print(x_test_source.shape)
x_test_values = x_test_source.drop(['Date Time','wd (deg)','wd_sin', 'wd_cos'], axis=1).values
x_test = split_x_stride(x_test_values, timesteps, 1)

# print(x.shape)
# print(y.shape)
# exit()
split_point = int(len(x) * 0.9)
x_train, x_val = x[split_point:], x[:split_point]
y_train, y_val = y[split_point:], y[:split_point]

#2. 모델구성
model = Sequential()
model.add(LSTM(100, input_shape=(24,18),activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(60, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(20, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(2))

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=10,verbose=1,
                   restore_best_weights= True)
model.compile(loss='mse', optimizer='adam')
filename = 'Keras56_2.hdf5'

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only= True,
    filepath=path+filename
)

model.fit(x,y,epochs= 10,validation_split=0.1, callbacks=[es,mcp,])

#4. 평가 예측
result = model.evaluate(x_val,y_val)
print('loss :', result)

y_pred = model.predict(x_test)

# print(y_pred)
# print(y_pred.shape)
pred_angle = np.degrees(np.arctan2(y_pred[:, 0], y_pred[:, 1])) % 360


def rmse(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print('Rmse :', rmse(y_test, pred_angle))
print(pred_angle.shape) 
print("예측값 (144개):", pred_angle.flatten())
date_times = train_csv1[
    (
    (train_csv1['Date Time'].dt.date == pd.to_datetime("2016-12-31").date()) |
    (train_csv1['Date Time'].dt.date == pd.to_datetime("2017-01-01").date())
    ) & (train_csv1['Date Time'] != "2016-12-31 00:00:00")
]['Date Time'].copy()
date_times = date_times.reset_index(drop=True)  # 인덱스 초기화

# y_pred가 numpy array라면 Series로 변환
y_pred_series = pd.Series(pred_angle.flatten(), name='wd (deg)')

# 3. 최종 결과 합치기
submission_df = pd.DataFrame({
    'Date Time': date_times,
    'wd (deg)': y_pred_series
})

# 4. CSV로 저장
submission_df.to_csv(path+"jena_홍석준_submit2.csv", index=False)
# jena_홍길동_submit.csv
