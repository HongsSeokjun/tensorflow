import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import time
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import r2_score
#1. 데이터
x1_datasets = np.array([range(100), range(301,401)]).T #(100,2)
                        # 삼성전자 종가, 하이닉스 종가
x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]).transpose()
#(100,3)                # 원유, 환율 , 금시세
y = np.array(range(2001, 2101))  # 화성의 화씨 온도
#(100,)

x_train1, x_test1, x_train2, x_test2, y_train, y_test =train_test_split(x1_datasets, x2_datasets, y, test_size=0.3, random_state=42)

#2-1 모델
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu', name='ibm1')(input1)
dense2 = Dense(50, activation='relu', name='ibm2')(dense1)
dense3 = Dense(30, activation='relu', name='ibm3')(dense2)
dense4 = Dense(40, activation='relu', name='ibm4')(dense3)
output1 = Dense(30, activation='relu', name='ibm5')(dense2) # 앙상블에서
# model1 = Model(inputs=input1,outputs=output1)

#2-2 모델
input2 = Input(shape=(3,))
dense21 = Dense(100, activation='relu', name='ibm21')(input2)
dense22 = Dense(50, activation='relu', name='ibm22')(dense21)
output2 = Dense(30, activation='relu', name='ibm23')(dense22)
dense22 = Dense(20, activation='relu', name='ibm22')(dense21)
output2 = Dense(10, activation='relu', name='ibm23')(dense22)
# model1 = Model(inputs=input2,outputs=output2)
# model1.summary()

#2-3 모델 합치기
from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([output1, output2], name='mg1')
merge2 = Dense(40,name='mg2')(merge1)
merge3 = Dense(20,name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input2], outputs = last_output)
# input을 merge1 넣으면 그냥 인풋한 시점에서 돌아가고 끝나서
# 최초의 input 을 넣어야 해당 Dense 층을 걸쳐 옴

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
hist = model.fit([x_train1,x_train2], y_train,epochs= 500,batch_size=4,verbose=2,validation_split=0.1,)# callbacks=[es,mcp],class_weight=class_weights,)

# 4. 평가 예측
loss = model.evaluate([x_test1, x_test2], y_test)

x1_pred = np.array([range(100,106), range(400,406)]).T
x2_pred = np.array([range(200,206), range(510,516),range(249,255)]).T
result = model.predict([x1_pred, x2_pred])

print('loss :',loss)
print('result :',result)
#y_pred는 2101 부터 2106까지
