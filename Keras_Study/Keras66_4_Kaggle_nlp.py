#https://www.kaggle.com/c/nlp-getting-started/overview

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout,BatchNormalization,Embedding,Conv1D,Flatten,Input
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import datetime
from tensorflow.keras.layers import concatenate, Concatenate
# 파일 저장을 위한 타임스탬프 경로 설정
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
path = 'C:\study25jun\_data\kaggle\Disaster\\'

train_csv = pd.read_csv(path+'train.csv',index_col=0)
test_csv =pd.read_csv(path+'test.csv', index_col=0)
submission_csv = pd.read_csv(path+'sample_submission.csv',index_col=0)


def merge_text(df):
    df['keyword'] = df['keyword'].fillna('unknown')
    df['location'] = df['location'].fillna('unknown').str.lower()
    df['text'] = df['text'] +' keyword: '+ df['keyword']+' location: ' +df['location']
    return df


# train_csv['keyword'] = train_csv['keyword'].astype(str)
# train_csv['location'] = train_csv['location'].astype(str)
# test_csv['keyword'] = test_csv['keyword'].astype(str)
# test_csv['location'] = test_csv['location'].astype(str)

# 고유한 값과 빈도 출력
# unique_keywords, counts = np.unique(train_csv['keyword'], return_counts=True)

# for keyword, count in zip(unique_keywords, counts):
#     print(f"Keyword: {keyword}, Count: {count}")

# print(np.unique(train_csv['keyword'], return_counts=True))
# print(train_csv['keyword'].shape) #(7613,)

train_csv = merge_text(train_csv)
test_csv = merge_text(test_csv)

y = train_csv['target']
x = train_csv.drop(columns=['target'])

x_train, x_test1, y_train, y_test1 = train_test_split(x,y,test_size=0.1,random_state=42)

# 3번째 컬럼 작업
x_test_csv_3col = test_csv['text'].values
x_train_3col = x_train['text'].values
x_test_3col = x_test1['text'].values

token = Tokenizer()
token.fit_on_texts(x) #icle': 979, 'charlotte': 980, 'daily': 981, 'guy': 982, 'jobs': 983, "th

x_train1_3col = token.texts_to_sequences(x_train_3col)
x_test1_3col = token.texts_to_sequences(x_test_3col)
x_pred_csv = token.texts_to_sequences(x_test_csv_3col)

# print(np.unique(y, return_counts=True)) #[0 1] #(array([0, 1], dtype=int64), array([4342, 3271], dtype=int64))

###### 패딩
from tensorflow.keras.preprocessing.sequence import pad_sequences
padding_x_3col = pad_sequences(x_train1_3col,
              padding='pre', # 'post' 디폴트는 프리
              maxlen=20, # 앞이 짤린다
              truncating='pre', # 'post' # 디폴트는 프리
              )
padding_pred_3col = pad_sequences(x_pred_csv,
                             padding='pre', # 'post' 디폴트는 프리
                            maxlen=20, # 앞이 짤린다
                            truncating='pre', # 'post' # 디폴트는 프리  
                             )
test_x_3col = pad_sequences(x_test1_3col,
              padding='pre', # 'post' 디폴트는 프리
              maxlen=20, # 앞이 짤린다
              truncating='pre', # 'post' # 디폴트는 프리
              )
# [x_test1, x_test2, x_test3]


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoder1 = OneHotEncoder(handle_unknown='ignore',sparse=False)
x_train_1col = x_train['keyword'].values
x_test_1col = x_test1['keyword'].values
x_test_csv_1col = test_csv['keyword'].values

x_train_1col = x_train_1col.reshape(-1,1)
x_train_1col = encoder.fit_transform(x_train_1col)

x_test_1col = x_test_1col.reshape(-1,1)
x_test_1col = encoder.transform(x_test_1col)

x_test_csv_1col = x_test_csv_1col.reshape(-1,1)
x_test_csv_1col = encoder.transform(x_test_csv_1col)

############### 2칼럼

x_train_2col = x_train['location'].values
x_test_2col = x_test1['location'].values
x_test_csv_2col = test_csv['location'].values

x_train_2col = x_train_2col.reshape(-1,1)
x_train_2col = encoder1.fit_transform(x_train_2col)

x_test_2col = x_test_2col.reshape(-1,1)
# print(x_test_2col.shape) #(6851, 2976)
# exit()
x_test_2col = encoder1.transform(x_test_2col)

x_test_csv_2col = x_test_csv_2col.reshape(-1,1)
x_test_csv_2col = encoder1.transform(x_test_csv_2col)

#2-1 모델
input1 = Input(shape=(222,))
dense1 = Dense(100, activation='relu', name='ibm1')(input1)
batch1 = BatchNormalization()(dense1)
drop1 = Dropout(0.1)(batch1)
output1 = Dense(50, activation='relu', name='ibm2')(drop1)

#2-2 모델
input2 = Input(shape=(2976,))
dense21 = Dense(100, activation='relu', name='ibm21')(input2)
batch2 = BatchNormalization()(dense21)
drop2 = Dropout(0.1)(batch2)
output2 = Dense(50, activation='relu', name='ibm22')(drop2)

#2-3 모델 구성
input3 = Input(shape=(20,))
Embed1 = Embedding(input_dim=1000, output_dim= 150,)(input3) #input_length=20)
batch3 = BatchNormalization()(Embed1)
drop3 = Dropout(0.1)(batch3)
conv1d1 = Conv1D(100, kernel_size=2,activation='relu')(drop3)
conv1d2 = Conv1D(50, kernel_size=2,activation='relu')(conv1d1)
batch4 = BatchNormalization()(conv1d2)
drop4 = Dropout(0.1)(batch4)
flatten = Flatten()(drop4)
output3 = Dense(20,  activation='relu')(flatten)

#2-4 모델 합치기

merge1 = concatenate([output1, output2,output2], name='mg1')
merge2 = Dense(40,name='mg2')(merge1)
batch5 = BatchNormalization()(merge2)
drop5 = Dropout(0.1)(batch5)
last_output = Dense(1, name='last')(drop5)

model = Model(inputs=[input1, input2,input3],outputs = last_output)

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=20,verbose=1,
                   restore_best_weights= True)
hist = model.fit([x_train_1col,x_train_2col,padding_x_3col],y, epochs= 50, batch_size= 6,verbose=2,validation_split=0.1, callbacks=[es,])

#4. 평가 예측
loss = model.evaluate([x_test_1col, x_test_2col, test_x_3col],y_test1,verbose=1)
result = model.predict([x_test_1col, x_test_2col, test_x_3col]) #원래의 y값과 예측된 y값의 비교
# print(result.shape) #(2246, 46)
# result = np.argmax(result, axis=1)
print('loss :',loss[0])
print('acc :',loss[1])

y_submit = model.predict([x_test_csv_1col,x_test_csv_2col,padding_pred_3col])
y_submit =  (y_submit > 0.5).astype(int)
submission_csv['target'] = y_submit

filename = path+f"submission_{timestamp}.csv"
submission_csv.to_csv(filename)
print(f"Submission saved to: {filename}")