from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
import pandas as pd

# train_datagen = ImageDataGenerator(
#     rescale=1./255, # 0~255 스케일링, 정규화
# )

# test_datagen = ImageDataGenerator(
#     rescale=1./255, # 0~255 스케일링, 정규화
# ) # 평가 데이터는 변환하지 않는다.
# path = './_data\\Kaggle\\cat_dog\\'
# path_train = 'C:\\Study25\\_data\\Kaggle\\cat_dog\\train2\\'
# path_test ='C:\\Study25\\_data\\Kaggle\\cat_dog\\test2\\'

# submission_csv = pd.read_csv(path+'sample_submission.csv',index_col=0)



# # 파일을 열꺼야
# xy_train = train_datagen.flow_from_directory(
#     path_train,             # 경로
#     target_size=(100,100),  # 리사이즈, 사이즈 규격 일치, 큰거는 축소, 작은거는 확대
#     batch_size=100,          # 160개의 사진을 (200,200,1) 16번으로 나누는 이유는 메모리 문제!
#     class_mode='binary',    # 
#     color_mode='rgb',
#     shuffle=True,
#     seed=333,
# ) # Found 160 images belonging to 2 classes. //2 classes = ad, normal

# import time

# xy_submit = test_datagen.flow_from_directory(
#     path_test,             # 경로
#     target_size=(100,100),  # 리사이즈, 사이즈 규격 일치, 큰거는 축소, 작은거는 확대
#     batch_size=100,          # 120개의 사진을 (200,200,1) 12번으로
#     class_mode='binary',    # 
#     color_mode='rgb',
#     shuffle=False,
# ) # Found 120 images belonging to 2 classes.
# # start = time.time()
# # x_train = xy_train[0][0]
# # y_train = xy_train[0][1]
# # x_test = xy_submit[0][0]
# # y_test =xy_submit[0][1]
# end = time.time()
# print(xy_train[0][0].shape) #(100, 100, 100, 3)
# print(xy_train[0][1].shape) #(100,)
# print(len(xy_train)) #250
# augment_size = 50
# randidx = np.random.randint(xy_train[0][0].shape[0], size=augment_size)
# print(randidx) #[14422 32911 45175 ... 18721 38642 34262]
# print(np.min(randidx), np.max(randidx)) # 0 59997

# x_augmented = xy_train[0][0][randidx].copy() # 4만개의 데이터 copy, copy로 새로운 메모리 할당.
# y_augmented = xy_train[0][1][randidx].copy()
# print(x_augmented)
# print(x_augmented.shape) #(50, 100, 100, 3)

# x_train = np.concatenate((xy_train[0][0], x_augmented))
# y_train = np.concatenate((xy_train[0][1], y_augmented))

# print(x_train.shape) #(150, 100, 100, 3)

# ######### 모든 수치화된 batch데이터를 하나로 합치기 #######
# all_x = []
# all_y = []
# for i in range(len(xy_train)):
#     x_batch, y_batch = xy_train[i]
#     all_x.append(x_batch)
#     all_y.append(y_batch)

# #print(all_x)
# all_x1 = []
# all_y1 = []
# for i in range(len(xy_submit)):
#     x_batch, y_batch = xy_submit[i]
#     all_x1.append(x_batch)
#     all_y1.append(y_batch)

# ###### 리스트를 하나의 numpy 배열로 합친다. ##### (사슬처럼 엮다.)
# x = np.concatenate(all_x, axis=0)
# y = np.concatenate(all_y, axis=0)

# x1 = np.concatenate(all_x1, axis=0)
# y1 = np.concatenate(all_y1, axis=0)
# # print('걸린 시간 :',round(end1- end,2),'초') #걸린 시간 : 47.7 초 걸린 시간 : 295.62 초
# print("x.shape :", x.shape) #x.shape : (25000, 200, 200, 3)
# print("y_shape :", y.shape) #y_shape : (25000,)
# start2 = time.time()
# np_path = 'C:\Study25\_data\_save_npy\\'
# np.save(np_path+'Keras50_catdog_x_train100.npy', arr=x)
# np.save(np_path+'Keras50_catdog_y_train100.npy', arr=y)

# np.save(np_path+'Keras50_catdog_x_test100.npy', arr=x1)
# np.save(np_path+'Keras50_catdog_y_test100.npy', arr=y1)

# end2 = time.time()

# print('걸린 시간 :',round(end2- start2,2),'초') # 걸린 시간 : 414.56 초 60.22 초
# exit()
np_path = 'C:\Study25\_data\_save_npy\\'
path = './_data\\Kaggle\\cat_dog\\'
submission_csv = pd.read_csv(path+'sample_submission.csv',index_col=0)

from sklearn.model_selection import train_test_split
#start = time.time()
x_train = np.load(np_path+"Keras50_catdog_x_train100.npy")
y_train = np.load(np_path+"Keras50_catdog_y_train100.npy")
x_test = np.load(np_path+"Keras50_catdog_x_test100.npy")
y_test = np.load(np_path+"Keras50_catdog_y_test100.npy")

print(x_train.shape, y_train.shape) #(25000, 100, 100, 3) (25000,)

x_train1, x_test1, y_train1, y_test1 = train_test_split(x_train,y_train,test_size=0.05, random_state= 47)

path = '.\_save\Keras44_catdog\\'
model = load_model(path+'k44_.hdf5')
#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=60,verbose=1,
                   restore_best_weights= True)


hist = model.fit(x_train1,y_train1, epochs= 200, batch_size= 30,verbose=2,validation_split=0.05, callbacks=[es])
#4. 평가 예측
loss = model.evaluate(x_test1,y_test1)
#result = model.predict(x_test1) #원래의 y값과 예측된 y값의 비교
print('loss :',loss[0])
print('acc :',loss[1])
y_submit = model.predict(x_test)
#y_pred =  (y_pred > 0.5).astype(int)
#images = x_test.reshape(-1, 200, 200)

submission_csv['label'] = y_submit

#################### csv파일 만들기 #########################
submission_csv.to_csv(path + 'submission_0617_17.csv') # CSV 만들기.

# loss : 0.5621001124382019
# acc : 0.7287999987602234

# loss : 0.5400758981704712
# acc : 0.7200000286102295