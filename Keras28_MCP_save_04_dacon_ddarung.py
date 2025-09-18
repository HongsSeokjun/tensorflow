import numpy as np # 전처리
import pandas as pd # 전처리 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
#print(np.__version__)#1.23.0
#print(pd.__version__)#2.2.3

# 1. 데이터

path='./_data/dacon/따릉이/'
train_csv = pd.read_csv(path+'train.csv', index_col=0)#. 현재 폴더 .. 이전 폴더
#print(train_csv) # [1459 rows x 11 columns] => [1459 rows x 10 columns]

test_csv = pd.read_csv(path+'test.csv', index_col=0)
#print(test_csv) #[715 rows x 9 columns]

submission_csv = pd.read_csv(path+'submission.csv', index_col=0)
#print(submission_csv) #[715 rows x 1 columns]

submission_ = pd.read_csv(path+'submission_0521_1400.csv',index_col=0)

#print(train_csv.shape) #(1459, 10)
#print(test_csv.shape) #(715, 9)
#print(submission_csv.shape) #(715, 1)

#print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object') Non- 결측치
#print(train_csv.info())

#print(train_csv.describe())
######################### 결측치 처리 1. 삭제 #############################
#print(train_csv.isnull().sum()) # 결측치의 개수 출력
#print(train_csv.isna().sum()) # 결측치의 개수 출력

train_csv = train_csv.dropna() # 결측치 제거 판다스 선처리 해야 함
#print(train_csv.isna().sum())
#print(train_csv.info())
#print(train_csv) #[1328 rows x 10 columns]

######################### 결측치 처리 2. 평균값 넣기 #############################

train_csv = train_csv.fillna(train_csv.mean()) #컬럼별 평균 [1459 rows x 9 columns]
# print(train_csv.isna().sum())
# print(train_csv.info())

######################### 테스트도 결측이 있다. #############################

#print(test_csv) 테이블이 밀릴 수 있어서 drpo 말고 mean으로 채워두기
test_csv = test_csv.fillna(test_csv.mean())
#print(test_csv.info())

x = train_csv.drop(['count'], axis= 1) # 행 또는 열 삭제
# count 라는 axis=1 열 삭제, 참고로 행 삭제는 axis = 0

y = train_csv['count']
#print(y.shape) #(1459,)

# features = ['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#             'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#             'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5']
# x = train_csv[features]

#print(x.hour.shape)
#exit()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state= 5917) #47 0.594 5917
# 2. 모델 구성
model = Sequential()
model.add(Dense(300, input_dim = 9, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
# 3,
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(
    monitor='val_loss', # 평가 지표로 확인하겠다
    mode= 'min', # 최대값 max, 알아서 찾아줘:auto
    patience=20, # 10번까지 초과해도 넘어가겠다
    restore_best_weights=True # val_loss값이 가장 낮은 값으로 저장 해놓겠다(False시 => 최소값 이후 10번째 값으로 그냥 잡는다.)
)
filename = '{epoch:04d}-{val_loss:.4f}Keras28_MCP_save_04_dacon_ddarung.hdf5'
path = '.\_save\Keras28_mcp\\04_dacon_ddarung\\'
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only= True,
    filepath=path+filename
)


hist = model.fit(x_train,y_train, epochs= 350, batch_size= 30,verbose=2,validation_split=0.1, callbacks=[es,mcp])

print(hist.history['loss'])
print(hist.history['val_loss'])

loss = model.evaluate(x_test,y_test)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교
r2 = r2_score(y_test, result)
print('loss :',loss)
print('R2 :',r2)

# submission.csv에 test_csv의 예측값 넣기
y_submit = model.predict(test_csv) # train데이터의 shape와 동일한 컬럼을 확인하고 넣어. x_train_shape:(N, 9)

#print('Rmse :',rmse(submission_,y_submit))
#print(y_submit.shape) # (715, 1)

######### submission.csv 파일 만들기 //count컬럼값만 넣어주기########
submission_csv['count'] = y_submit
#print(submission_csv)


##################### csv파일 만들기 #########################
submission_csv.to_csv(path + 'submission_0526_1830.csv') # CSV 만들기.

#print('x의 예측값 :',result)

# default Epoch 100
# loss : 2661.24365234375
# R2 : 0.6286171463977811
#val_loss 3396.74658203125

# EarlyStopping, restore_best_weights=True, patience=10 Epoch 29/100
# loss : 2634.500244140625
# R2 : 0.6323492767487997
# val_loss 3131.449462890625

# EarlyStopping, restore_best_weights=False, patience=10  Epoch 22/100
# loss : 3951.268310546875
# R2 : 0.4485911992913326
# val_loss 4673.65478515625

# loss : 2658.750732421875
# R2 : 0.6289650549504096