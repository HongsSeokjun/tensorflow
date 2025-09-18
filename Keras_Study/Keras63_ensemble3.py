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
x3_datasets = np.array([range(100), range(301,401), range(77,177), range(33,133)]).transpose()

y1 = np.array(range(2001, 2101))  # 화성의 화씨 온도
y2 = np.array(range(13001,13101))
#(100,)

x_train1, x_test1, x_train2, x_test2, x_train3, x_test3, y_train1, y_test1,y_train2,y_test2 =train_test_split(x1_datasets, x2_datasets,x3_datasets, y1,y2, test_size=0.3, random_state=44)

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
dense23 = Dense(30, activation='relu', name='ibm23')(dense22)
dense24 = Dense(20, activation='relu', name='ibm24')(dense23)
output2 = Dense(10, activation='relu', name='ibm25')(dense24)
# model1 = Model(inputs=input2,outputs=output2)
# model1.summary()

#2-3 모델
input3 = Input(shape=(4,))
dense31 = Dense(100, activation='relu', name='ibm31')(input3)
dense32 = Dense(50, activation='relu', name='ibm32')(dense31)
dense33 = Dense(30, activation='relu', name='ibm33')(dense32)
dense34 = Dense(20, activation='relu', name='ibm34')(dense33)
output3 = Dense(10, activation='relu', name='ibm35')(dense34)

#2-3 모델 합치기
from tensorflow.keras.layers import concatenate, Concatenate
merge1 = Concatenate(axis=1)([output1, output2,output3]) #  가로는 클래스란걸 명시
merge2 = Dense(40,name='mg2')(merge1)
merge3 = Dense(20,name='mg3')(merge2)
middle_output = Dense(1, name='output1')(merge3)
# last_output1 = Dense(1, name='output1')(merge3)  # 첫 번째 출력 (y1)



#2-4 분리1 -> y1
last_output11 = Dense(10,name='last11')(middle_output)
last_output12 = Dense(10,name='last12')(last_output11)
last_output13 = Dense(1,name='last13')(last_output12)

#2-5 분리2 -> y2
last_output21 = Dense(1,name='last21')(middle_output)

# merge4 = Concatenate(axis=1)([output1, output2,output3]) #  가로는 클래스란걸 명시
# merge5 = Dense(40,name='mg4')(merge4)
# merge6 = Dense(20,name='mg5')(merge5)
# last_output2 = Dense(1, name='output2')(merge6)  # 두 번째 출력 (y2)

model = Model(inputs=[input1,input2,input3], outputs = [last_output13,last_output21])
# input을 merge1 넣으면 그냥 인풋한 시점에서 돌아가고 끝나서
# 최초의 input 을 넣어야 해당 Dense 층을 걸쳐 옴

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
hist = model.fit([x_train1,x_train2,x_train3], [y_train1,y_train2],epochs= 100,batch_size=2,verbose=2,validation_split=0.1,)# callbacks=[es,mcp],class_weight=class_weights,)

# 4. 평가 예측
loss = model.evaluate([x_test1, x_test2, x_test3], [y_test1,y_test2])
print('loss :',loss) #loss : [43.745704650878906, 29.63214874267578, 14.113556861877441, 4.500187397003174, 1.2226887941360474]
x1_pred = np.array([range(100,106), range(400,406)]).T
x2_pred = np.array([range(200,206), range(510,516),range(249,255)]).T
x3_pred = np.array([range(100,106), range(400,406),range(177,183),range(133,139)]).T

result = model.predict([x1_pred,x2_pred,x3_pred])
print(result)

#파이썬 기초 줄바꿈 \