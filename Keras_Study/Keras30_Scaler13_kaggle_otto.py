# 다중분류
#https://www.kaggle.com/competitions/otto-group-product-classification-challenge/datafrom sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
# 1. 데이터
path ='C:\Study25\_data\Kaggle\Otto\\'

train_csv = pd.read_csv(path+'train.csv',index_col=0)
test_csv =pd.read_csv(path+'test.csv', index_col=0)
submission_csv = pd.read_csv(path+'sampleSubmission.csv',index_col=0)
#print(train_csv.columns)
# Index(['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7',
#        'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13',
#        'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19',
#        'feat_20', 'feat_21', 'feat_22', 'feat_23', 'feat_24', 'feat_25',
#        'feat_26', 'feat_27', 'feat_28', 'feat_29', 'feat_30', 'feat_31',
#        'feat_32', 'feat_33', 'feat_34', 'feat_35', 'feat_36', 'feat_37',
#        'feat_38', 'feat_39', 'feat_40', 'feat_41', 'feat_42', 'feat_43',
#        'feat_44', 'feat_45', 'feat_46', 'feat_47', 'feat_48', 'feat_49',
#        'feat_50', 'feat_51', 'feat_52', 'feat_53', 'feat_54', 'feat_55',
#        'feat_56', 'feat_57', 'feat_58', 'feat_59', 'feat_60', 'feat_61',
#        'feat_62', 'feat_63', 'feat_64', 'feat_65', 'feat_66', 'feat_67',
#        'feat_68', 'feat_69', 'feat_70', 'feat_71', 'feat_72', 'feat_73',
#        'feat_74', 'feat_75', 'feat_76', 'feat_77', 'feat_78', 'feat_79',
#        'feat_80', 'feat_81', 'feat_82', 'feat_83', 'feat_84', 'feat_85',
#        'feat_86', 'feat_87', 'feat_88', 'feat_89', 'feat_90', 'feat_91',
#        'feat_92', 'feat_93', 'target']
#print(train_csv.shape)(61878, 94)
x = train_csv.drop(['target'],axis=1)
y = train_csv['target']
#print(np.unique(y, return_counts = True))
#(array(['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6',
#       'Class_7', 'Class_8', 'Class_9'], dtype=object), array([ 1929, 16122,  8004,  2691,  2739, 14135,  2839,  8464,  4955],
y = pd.get_dummies(y)
#print(x.shape)#(61878, 93)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state= 47,stratify=y)
#print(y_train.values)
#labels = np.argmax(y_train.values, axis=1)
# print(labels)
#print(np.unique(labels, return_counts=True))
standard = StandardScaler() #표준화
x_train= standard.fit_transform(x_train)        # train 데이터에 맞춰서 스케일링
x_test= standard.transform(x_test) # test 데이터는 transform만!

#2. 모델 구성
model = Sequential([
    Dense(256, input_dim=93, activation='relu'),
    BatchNormalization(),
    (Dropout(0.3)), #드랍아웃은 layer 깊이가 있을때 사용
    Dense(126, activation='relu'),
    BatchNormalization(),
    (Dropout(0.3)),
    Dense(62, activation='relu'),
    BatchNormalization(),
    (Dropout(0.3)),
    Dense(30, activation='relu'),
    BatchNormalization(),
    (Dropout(0.3)),
    Dense(16, activation='relu'),
    BatchNormalization(),
    (Dropout(0.3)),
    Dense(y.shape[1], activation='softmax')
])

#3. 컴파일, 훈련
labels = np.argmax(y_train.values, axis=1)
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels) ,
    y =labels
)
class_weights = dict(enumerate(weights))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True,
)

filename = 'Keras30_Scaler13_kaggle_otto.hdf5'
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only= True,
    filepath=path+filename
)

model.fit(x_train,y_train,epochs=400, batch_size=32,verbose=2,validation_split=0.1,callbacks=[es,mcp],class_weight=class_weights,)

#4. 평가,훈련
loss = model.evaluate(x_test, y_test)
print('loss :', loss[0])
print('acc :',loss[1])
y_predict = model.predict(x_test)
#y_predict =  y_predict.round(1)


y_submit = model.predict(test_csv)
#print(y_submit)
#y_submit = y_submit.round(1)
print(y_submit)
submission_csv[['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']] = y_submit

#################### csv파일 만들기 #########################
submission_csv.to_csv(path + 'submission_0609_17.csv') # CSV 만들기.
