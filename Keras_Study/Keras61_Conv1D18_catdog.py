import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D,Dense, Dropout,BatchNormalization,Conv1D,Flatten
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import matplotlib.pylab as plt
import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split
np_path = 'C:\study25ju\_data\_save_npy\\'
path = './_data\\Kaggle\\cat_dog\\'
# submission_csv = pd.read_csv(path+'sample_submission.csv',index_col=0)

#start = time.time()
x_train = np.load(np_path+"keras44_01_x_train.npy")
y_train = np.load(np_path+"Keras44_01_y_train.npy")
#end = time.time()
print(x_train.shape, y_train.shape) #(25000, 100, 100, 3) (25000,)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],x_train.shape[2]*x_train.shape[3]) #(50000, 3072)


print(x_train.shape) #(25000, 100, 300)
# exit()
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_train,y_train,test_size=0.5, random_state= 47)

#print(x_train1.shape)(22500, 100, 100, 3)
#print(x_test1.shape) (2500, 100, 100, 3)
#exit()
#2. 모델 구성
model = Sequential()
model.add(Conv1D(256, kernel_size=2,input_shape=(100,300), activation='relu'))
model.add(Conv1D(30,2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(16,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(8,  activation='relu'))#XGBoost가 받아야 하는 feature 수는 훈련할 때의 입력 특성 수와 동일해야 합니다.
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=60,verbose=1,
                   restore_best_weights= True)



hist = model.fit(x_train1,y_train1, epochs= 1, batch_size= 30,verbose=2,validation_split=0.05, callbacks=[es,])
#4. 평가 예측
loss = model.evaluate(x_test1,y_test1)
#result = model.predict(x_test1) #원래의 y값과 예측된 y값의 비교
print('loss :',loss[0])
print('acc :',loss[1])
# y_submit = model.predict(x_test)
# #y_pred =  (y_pred > 0.5).astype(int)
# #images = x_test.reshape(-1, 200, 200)

# submission_csv['label'] = y_submit

# #################### csv파일 만들기 #########################
# submission_csv.to_csv(path + 'submission_0613_17_2.csv') # CSV 만들기.

# import matplotlib
# matplotlib.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize=(12, 4))
# for i in range(10):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(images[i],cmap='gray')
#     plt.title(f"예측:{y_pred[i]} / 정답:{y_test[i]}")
#     plt.axis('off')
# plt.tight_layout()
# plt.show()

# loss : 0.5400758981704712
# acc : 0.7200000286102295

# loss : 0.669147253036499
# acc : 0.5911999940872192