from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from tensorflow.keras.models import Sequential,load_model
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
datasets = fetch_covtype()

x = datasets.data
y = datasets.target
print(x)
exit()
#print(x.shape, y.shape) # (581012, 54) (581012,)
#print(np.unique(y, return_counts = True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
#print(pd.value_counts(y))

# encorder = OneHotEncoder(sparse=False)
# y = y.reshape(-1,1)
# y = encorder.fit_transform(y)

y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.05, random_state= 47)#,stratify=y)
#print(y_train.values)
#labels = np.argmax(y_train.values, axis=1)
# print(labels)
#print(np.unique(labels, return_counts=True))
standard = StandardScaler() #표준화
x_train= standard.fit_transform(x_train)        # train 데이터에 맞춰서 스케일링
x_test= standard.transform(x_test) # test 데이터는 transform만!

path = '.\_save\Keras28_mcp\\10_fetch_covtype\\'
model = load_model(path+'0002-0.4911Keras28_MCP_save_10_fetch_covtype.hdf5')
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
#4. 평가,훈련
loss = model.evaluate(x_test, y_test)
print('loss :', loss[0])
print('acc :',loss[1])
y_predict = model.predict(x_test)

# loss : 0.4877546429634094
# acc : 0.7962548732757568