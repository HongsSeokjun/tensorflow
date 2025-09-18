from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization,LSTM,Flatten,Conv2D
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
plt.imshow(datasets.images[9], cmap='gray')
# plt.title(f"Label: {datasets.target[9]}")
# plt.show()
# print(y)
# exit()
print(np.max(x), np.min(x))

x = x.reshape(-1, 8, 8)  # (batch, height, width, channel)
# print(x.shape)
# exit()
y = pd.get_dummies(y)

print(y.columns)
print(y.sum())
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state= 47,stratify=y)
# exit()

#2. 모델구성
model = Sequential()
model.add(LSTM(32,input_shape=(8,8), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
# model.add(Conv2D(16,(2,2), activation='relu'))
# model.add(Flatten())
model.add(Dense(8,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
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

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
hist = model.fit(x_train, y_train,epochs= 1, batch_size= 4,verbose=2,validation_split=0.1,class_weight=class_weights,)# callbacks=[mcp])


# 4. 평가 예측

loss = model.evaluate(x_test,y_test)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교
y_pred = np.argmax(result, axis=1)
# 실제 정답도 같은 형식으로 바꿔야 비교 가능
y_true = np.argmax(y_test.values, axis=1)

# r2 = r2_score(y_test, result)
print('loss :',loss)
# 예시 출력
print("예측값 :", y_pred[:10])
print("실제값 :", y_true[:10])

# 이미지 데이터는 원래 shape (8, 8)이므로 다시 꺼내기
images = x_test.reshape(-1, 8, 8)

plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"예측:{y_pred[i]} / 정답:{y_true[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# loss : [1.9051568508148193, 0.28333333134651184]