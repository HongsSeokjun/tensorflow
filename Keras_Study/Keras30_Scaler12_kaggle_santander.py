#데이터가 불균형, 이진
#https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization,LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,OneHotEncoder,StandardScaler
from sklearn.utils import class_weight

# 1. 데이터
path = 'C:\Study25\_data\Kaggle\santander\\'

train_csv = pd.read_csv(path+'train.csv',index_col=0)
test_csv =pd.read_csv(path+'test.csv', index_col=0)
submission_csv = pd.read_csv(path+'sample_submission.csv',index_col=0)

print(train_csv)
#(200000, 201)
# Index(['target', 'var_0', 'var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6',
#        'var_7', 'var_8',
#        ...
#        'var_190', 'var_191', 'var_192', 'var_193', 'var_194', 'var_195',
#        'var_196', 'var_197', 'var_198', 'var_199'],
#       dtype='object', length=201)
x = train_csv.drop(['target'],axis=1)
y = train_csv['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state= 57,stratify=y)

standard = StandardScaler() #표준화
scaler = MinMaxScaler() # 정규화
x_train = standard.fit_transform(x_train)        # train 데이터에 맞춰서 스케일링
x_test= standard.transform(x_test) # test 데이터는 transform만!
test_csv = standard.transform(test_csv)

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y =y_train
)
class_weights = dict(enumerate(weights))

#2. 모델 구조
model = Sequential()
model.add(Dense(256, input_dim=200, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128,   activation='relu'))#activation=LeakyReLU(alpha=0.1)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64,   activation='relu'))#activation=LeakyReLU(alpha=0.1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(8,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일 ,훈련
es = EarlyStopping(
    monitor='val_loss', # 지표를 acc로 잡으면 max로 잡아할때도 있다. => auto로 잡으면 알아서 잡아줌
    mode='min',
    patience= 150,
    restore_best_weights=True,
)

filename = 'Keras30_Scaler12_kaggle_santander.hdf5'
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only= True,
    filepath=path+filename
)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])#categorical_crossentropy 
hist = model.fit(x_train, y_train, epochs = 600, batch_size=32, verbose=2, validation_split=0.1, callbacks=[es,mcp],class_weight=class_weights,)

# 4. 평가 예측
results = model.evaluate(x_test, y_test)
print(results)

y_predict = model.predict(x_test)
y_predict =  (y_predict > 0.5).astype(int)
f1_score1 = f1_score(y_test, y_predict)
print('f1_score :', f1_score1)

y_submit = model.predict(test_csv)
y_submit =  (y_submit > 0.5).astype(int)
submission_csv['target'] = y_submit

#################### csv파일 만들기 #########################
submission_csv.to_csv(path + 'submission_0609_18.csv') # CSV 만들기.

import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

plt.figure(figsize=(9,6)) # 9 x 6
plt.plot(hist.history['loss'], c = 'red', label = 'loss') # x축은 epochs, y값만 넣으면 시간순으로 그림 그림
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss')
plt.plot(hist.history['acc'], c = 'green', label = 'acc')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right') # 우측 상단에 라벨 표시
plt.grid() # 격자 표시
plt.show()