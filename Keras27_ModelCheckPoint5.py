#Keras27_ModelCheckPoint3 복붙
import sklearn as sk
print(sk.__version__) #0.24.2 #1.1.3
from sklearn.datasets import load_boston
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
from sklearn.preprocessing import StandardScaler,MinMaxScaler
#1. 데이터
dataset = load_boston()
#print(dataset)
#print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

x = dataset.data
y = dataset.target
#print(x.shape, y.shape) #(506, 13) (506,)
#print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state=42)

scaler = StandardScaler()
minmaxs = MinMaxScaler()
x_train = minmaxs.fit_transform(x_train)
x_test = minmaxs.transform(x_test)

# print(np.min(x_train), np.max(x_train))
# print(np.min(x_test), np.max(x_test))

#2. 모델 구성

model = Sequential([
    Dense(32, input_dim = 13, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1)
])

#model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor='val_loss', # 평가 지표로 확인하겠다
    mode= 'min', # 최대값 max, 알아서 찾아줘:auto
    patience=10, # 10번까지 초과해도 넘어가겠다
    restore_best_weights=True # val_loss값이 가장 낮은 값으로 저장 해놓겠다(False시 => 최소값 이후 10번째 값으로 그냥 잡는다.)
)
############### mcp 세이브 파일명 만들기 ##############
import datetime
date = datetime.datetime.now()
print(date)
print(type(date))
date = date.strftime('%m%d_%H%M%S')
print(date)
print(type(date)) # <class 'str'>

path = '.\_save\Keras27_mcp2\\'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path,'k27_',date,'_',filename])
#print(filepath)
#exit()

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only= True,
    filepath=filepath
)


hist = model.fit(x_train,y_train, epochs= 100, batch_size= 3,verbose=2,validation_split=0.1, callbacks=[es,mcp])

#4. 평가 예측
loss = model.evaluate(x_test,y_test)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교
r2 = r2_score(y_test, result)
print('loss :',loss)
print('R2 :',r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test,result)
print('Rmse :',rmse)

