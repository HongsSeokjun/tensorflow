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

y1 = np.array(range(2001, 2101))  # 화성의 화씨 온도
y2 = np.array(range(13001,13101))
#(100,)

x_train1, x_test1, y_train1, y_test1,y_train2,y_test2 =train_test_split(x1_datasets, y1,y2, test_size=0.3, random_state=44)

#2. 모델
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu',name = 'ibm1')(input1)
dense2 = Dense(50, activation='relu', name = 'ibm2')(dense1)
dense3 = Dense(30, activation='relu',name = 'ibm3')(dense2)
output1 = Dense(30, activation='relu', name = 'ibm5')(dense3)

#2-1 y값 분리1 -> y1
last_output11 = Dense(10,name='last11')(output1)
last_output12 = Dense(10,name='last12')(last_output11)
last_output13 = Dense(1,name='last13')(last_output12)

#2-2 y값 분리2 -> y2
last_output21 = Dense(10,name='last21')(output1)
last_output22 = Dense(1,name='last22')(last_output21)

model = Model(inputs= input1, outputs = [last_output13,last_output22])
# input을 merge1 넣으면 그냥 인풋한 시점에서 돌아가고 끝나서
# 최초의 input 을 넣어야 해당 Dense 층을 걸쳐 옴

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
hist = model.fit(x_train1, [y_train1,y_train2],epochs= 100,batch_size=2,verbose=2,validation_split=0.1,)# callbacks=[es,mcp],class_weight=class_weights,)

# 4. 평가 예측
loss = model.evaluate(x_test1, [y_test1,y_test2])
print('loss :',loss) #loss : [43.745704650878906, 29.63214874267578, 14.113556861877441, 4.500187397003174, 1.2226887941360474]
x1_pred = np.array([range(100,106), range(400,406)]).T
# x2_pred = np.array([range(200,206), range(510,516),range(249,255)]).T
# x3_pred = np.array([range(100,106), range(400,406),range(177,183),range(133,139)]).T

result = model.predict(x1_pred)
print(result)
# loss : [1241.5965576171875, 1214.5350341796875, 27.06163787841797, 28.763599395751953, 3.9989256858825684]
# [array([[2036.7721],
#        [2038.6029],
#        [2040.4337],
#        [2042.264 ],
#        [2044.1028],
#        [2045.9485]], dtype=float32), array([[13090.531],
#        [13102.338],
#        [13114.146],
#        [13125.953],
#        [13137.828],
#        [13149.76 ]], dtype=float32)]