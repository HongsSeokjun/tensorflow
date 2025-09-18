from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
np_path = 'C:\Study25\_save\keras46_horse\\'

x_train = np.load(np_path+"Keras46_03_x_train100.npy")
y_train = np.load(np_path+"Keras46_03_y_train100.npy")


#print(x_train.shape, y_train.shape)  #(1027, 100, 100, 3) (1027,)

augment_size = 1000  # 증가시킬 사이즈 

############# 요기부터 증폭 ###################
datagen = ImageDataGenerator(
    # rescale=1./255, # 0~255 스케일링, 정규화
    horizontal_flip=True, # 수평 뒤집기 <- 데이터 증폭 또는 변환
    # vertical_flip= True, # 수직 뒤집기 <- 데이터 증폭 또는 변환
    width_shift_range=0.1, # 평행이동 10%
    # height_shift_range=0.1, # 수직이동 10%
    # rotation_range=15, # 회전 5도
    # zoom_range=1.2, # 줌 1.2배
    # shear_range=0.7, # 좌표하나를 고정시키고, 다른 몇개의 좌표를 이동시키는 변환(찌부 만들기)
    # fill_mode='nearest',
)

randidx = np.random.randint(x_train.shape[0], size=augment_size)

# print(randidx) #[14422 32911 45175 ... 18721 38642 34262]
# print(np.min(randidx), np.max(randidx)) # 0 59997

x_augmented = x_train[randidx].copy() # 4만개의 데이터 copy, copy로 새로운 메모리 할당.
                                      # 서로 영향 x
y_augmented = y_train[randidx].copy()
# print(x_augmented)
# print(x_augmented.shape) #(500, 100, 100, 3)

x_augmented = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    save_to_dir='C:\Study25\_data\_save_img\\06_horse\\',
    save_prefix='horse',     # 파일 이름 앞부분 설정 가능
    shuffle=False,
).next()[0]

print(x_augmented.shape) #(40000, 28, 28, 1)

x_train = x_train.reshape(-1,100,100,3)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape) #(1527, 100, 100, 3)
print(y_train.shape) #(1527,)

x_train1, x_test, y_train1, y_test = train_test_split(x_train,y_train,test_size=0.1, random_state= 47)


path = '.\_save\keras46_horse\\'
model = load_model(path+'k46_4.hdf5')

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=60,verbose=1,
                   restore_best_weights= True)


#####################################################

hist = model.fit(x_train,y_train, epochs= 250, batch_size= 4,verbose=2,validation_split=0.1, callbacks=[es])
#4. 평가 예측
loss = model.evaluate(x_test,y_test)
result = model.predict(x_test) #원래의 y값과 예측된 y값의 비교
print('loss :',loss[0])
print('acc :',loss[1])
y_pred = model.predict(x_test)
y_pred =  (y_pred > 0.5).astype(int)
images = x_test.reshape(-1, 100, 100,3)

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i],cmap='gray')
    plt.title(f"예측:{y_pred[i]} / 정답:{y_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# loss : 0.007188461255282164
# acc : 1.0