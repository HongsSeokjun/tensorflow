#https://www.kaggle.com/c/nlp-getting-started/overview

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,BatchNormalization,Embedding,Conv1D,Flatten
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import datetime

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
train_csv = merge_text(train_csv)
test_csv = merge_text(test_csv)

x = train_csv['text'].values
y = train_csv['target']
x_test_csv = test_csv['text'].values

token = Tokenizer()
token.fit_on_texts(x) #icle': 979, 'charlotte': 980, 'daily': 981, 'guy': 982, 'jobs': 983, "th

x = token.texts_to_sequences(x)
x_test_csv = token.texts_to_sequences(x_test_csv)

print('뉴스기사의 최대길이 :', max(len(i) for i in x)) # 40
print('뉴스기사의 최소길이 :', min(len(i) for i in x))
print('뉴스기사의 평균길이 :', sum(map(len, x))/len(x)) #21.79114672271115
# print(np.unique(y, return_counts=True)) #[0 1] #(array([0, 1], dtype=int64), array([4342, 3271], dtype=int64))
###### 패딩
from tensorflow.keras.preprocessing.sequence import pad_sequences
padding_x = pad_sequences(x,
              padding='pre', # 'post' 디폴트는 프리
              maxlen=20, # 앞이 짤린다
              truncating='pre', # 'post' # 디폴트는 프리
              )
padding_pred = pad_sequences(x_test_csv,
                             padding='pre', # 'post' 디폴트는 프리
                            maxlen=20, # 앞이 짤린다
                            truncating='pre', # 'post' # 디폴트는 프리  
                             )

# print(padding_x.shape) #(7613, 20)
x_train, x_test1, y_train, y_test1 = train_test_split(padding_x,y,test_size=0.1,random_state=42)

#2. 모델 구성
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim= 150, input_length=20))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Conv1D(100, kernel_size=2,activation='relu'))
model.add(Conv1D(50, kernel_size=2,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(20,  activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=20,verbose=1,
                   restore_best_weights= True)
hist = model.fit(padding_x,y, epochs= 50, batch_size= 6,verbose=2,validation_split=0.1, callbacks=[es,])

#4. 평가 예측
loss = model.evaluate(x_test1,y_test1,verbose=1)
result = model.predict(x_test1) #원래의 y값과 예측된 y값의 비교
# print(result.shape) #(2246, 46)
# result = np.argmax(result, axis=1)
print('loss :',loss[0])
print('acc :',loss[1])

y_submit = model.predict(padding_pred)
y_submit =  (y_submit > 0.5).astype(int)
submission_csv['target'] = y_submit

filename = path+f"submission_{timestamp}.csv"
submission_csv.to_csv(filename)
print(f"Submission saved to: {filename}")