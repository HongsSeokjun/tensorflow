from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten , Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score#불균형 데이터일때
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pylab as plt
from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint
#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
#print(x.shape) #(1797, 64)
#print(y.shape) #(1797,)
# plt.imshow(datasets.images[9], cmap='gray')
# plt.title(f"Label: {datasets.target[9]}")
# plt.show()
print(y)
exit()

y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state= 47,stratify=y)
# standard = StandardScaler() #표준화
# x_train= standard.fit_transform(x_train)        # train 데이터에 맞춰서 스케일링
# x_test= standard.transform(x_test) # test 데이터는 transform만!

#2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64,   activation='relu'))#activation=LeakyReLU(alpha=0.1)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(32,   activation='relu'))#activation=LeakyReLU(alpha=0.1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(16,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

#3. 컴파일, 훈련
labels = np.argmax(y_train.values, axis=1)
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y =labels
)
class_weights = dict(enumerate(weights))

# path = 'C:\Study25\_save\Keras30_mcp\\'
# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     save_best_only= True,
#     filepath=path+"digits"
# )


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
hist = model.fit(x_train, y_train,epochs= 300, batch_size= 4,verbose=2,validation_split=0.1,class_weight=class_weights)#, callbacks=[mcp])


# 4. 평가 예측

loss = model.evaluate(x_test,y_test)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교
result =  (result > 0.5).astype(int)
r2 = r2_score(y_test, result)
print('loss :',loss)
print('R2 :',r2)
#first_col = result[:, 0]
print(result[0])
