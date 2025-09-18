from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
import pandas as pd
(x_train, y_train), (x_test, y_test) =  mnist.load_data()

x_train = x_train/255.
x_test = x_test/255.

augment_size = 10000  # 증가시킬 사이즈 
############# 요기부터 증폭 ###################
datagen = ImageDataGenerator(
    rescale=1./255, # 0~255 스케일링, 정규화
    width_shift_range=0.1, # 평행이동 10%
)


randidx = np.random.randint(x_train.shape[0], size=augment_size)
# => 같은 내용 np.random.randint(60000, 40000) 0~ 60000의 숫자가 4만개

print(randidx) #[14422 32911 45175 ... 18721 38642 34262]
print(np.min(randidx), np.max(randidx)) # 0 59997

x_augmented = x_train[randidx].copy() # 4만개의 데이터 copy, copy로 새로운 메모리 할당.
                                      # 서로 영향 x
y_augmented = y_train[randidx].copy()
print(x_augmented)
print(x_augmented.shape) #(40000, 28, 28)

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2],
    1,
)

print(x_augmented.shape) #(40000, 28, 28, 1)

x_augmented = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    save_to_dir='C:\Study25\_data\_save_img\\02_mnist\\',
    save_prefix='mnist',     # 파일 이름 앞부분 설정 가능
    shuffle=False,
).next()[0]
exit()
print(x_augmented.shape) #(40000, 28, 28, 1)

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

print(x_train.shape)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape,y_train.shape)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


path = '.\_save\Keras36_cnn5\\'
model = load_model(path+'k36_0610_175717_0040-0.0251.hdf5')
#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=50,verbose=1,
                   restore_best_weights= True)
hist = model.fit(x_train,y_train, epochs= 500, batch_size= 512,verbose=2,validation_split=0.1, callbacks=[es])
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])

loss = model.evaluate(x_test,y_test,verbose=1)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교
print('loss :',loss[0])
print('acc :',loss[1])
y_test = y_test.values  #=> 판다스를 넘파이로
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test =  np.argmax(y_test, axis=1)



# gpu
# loss : 2.3010332584381104
# acc : 0.11349999904632568
# time : 312.3143141269684
# gpu
# loss : 0.034305837005376816
# acc : 0.9923999905586243
# time : 223.3766438961029

# loss : 0.02285638079047203
# acc : 0.9944999814033508