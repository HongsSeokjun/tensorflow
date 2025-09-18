from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pylab as plt
augment_size = 100  # 증가시킬 사이즈 

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)
print(x_train[0].shape) #(28, 28)

# plt.imshow(x_train[0],cmap='gray')
# plt.show()

aaa = np.tile(x_train[0].reshape(28*28),augment_size).reshape(-1,28,28,1) # 1차원 벡터 형태로 만든 뒤에 증폭(tile)하는 것이 가장 안전합니다.
# print(aaa.shape)  #(100, 28, 28, 1)

############# 요기부터 증폭 ###################
datagen = ImageDataGenerator(
    rescale=1./255, # 0~255 스케일링, 정규화
    horizontal_flip=True, # 수평 뒤집기 <- 데이터 증폭 또는 변환
    # vertical_flip= True, # 수직 뒤집기 <- 데이터 증폭 또는 변환
    width_shift_range=0.1, # 평행이동 10%
    # height_shift_range=0.1, # 수직이동 10%
    # rotation_range=15, # 회전 5도
    # zoom_range=1.2, # 줌 1.2배
    # shear_range=0.7, # 좌표하나를 고정시키고, 다른 몇개의 좌표를 이동시키는 변환(찌부 만들기)
    fill_mode='nearest',
)

# 파일을 열꺼야
xy_data = datagen.flow(
    #np.tile(x_train[0].reshape(28*28),augment_size).reshape(-1,28,28,1),    # x데이터
    aaa,
    np.zeros(augment_size), # y데이터 생성, 전부 0으로 가득찬 y값.
    batch_size=augment_size,
    shuffle=False,
).next()

x_data, y_data = xy_data

print(x_data.shape)
print(y_data.shape)
print(xy_data)
print(len(xy_data))


exit()
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.imshow(x_data[i],cmap='gray')
plt.show()