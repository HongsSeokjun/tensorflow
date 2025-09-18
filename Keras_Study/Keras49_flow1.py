from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img # 이미지 땡겨오기.
from tensorflow.keras.preprocessing.image import img_to_array # 땡겨온 이미지 수치화
import numpy as np
import matplotlib.pylab as plt
from tensorflow.keras.models import load_model
path = 'C:\Study25\_data\image\me\\'

img = load_img(path+'me.jpg', target_size=(100,100),)
print(img)
print(type(img))#<class 'PIL.Image.Image'>
#PIL = Python Image Library

# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
# print(arr)
# print(arr.shape) #(250, 250, 3)
# arr = arr.reshape(1,250,250,3)
# print(arr.shape)
img = np.expand_dims(arr, axis=0)
print(arr.shape)
 
# img = img /255
# print(img)
#np.save(path+'Keras47_me_catdog.npy',arr=img)

############# 요기부터 증폭 ###################

datagen = ImageDataGenerator(
    rescale=1./255, # 0~255 스케일링, 정규화
    horizontal_flip=True, # 수평 뒤집기 <- 데이터 증폭 또는 변환
    # vertical_flip= True, # 수직 뒤집기 <- 데이터 증폭 또는 변환
    width_shift_range=0.1, # 평행이동 10%
    # height_shift_range=0.1, # 수직이동 10%
    rotation_range=15, # 회전 5도
    # zoom_range=1.2, # 줌 1.2배
    # shear_range=0.7, # 좌표하나를 고정시키고, 다른 몇개의 좌표를 이동시키는 변환(찌부 만들기)
    fill_mode='nearest',
)

# 파일을 열꺼야
it = datagen.flow(
    img,
    batch_size=1,
)
print('=========================================================================')
print(it) #<keras.preprocessing.image.NumpyArrayIterator object at 0x000001C2ED7F64C0>
print('=========================================================================')
# print(it.next())
# aaa = it.next() # 파이썬 2.0 문법
# print(aaa)
# print(aaa.shape)#(1, 100, 100, 3)

# bbb = next(it) next의 다음꺼 출력
# print(bbb)
# print(bbb.shape) #(1, 100, 100, 3)

# print(it.next()) #
# print(it.next())
# print(it.next())

fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(5,5))
ax = ax.flatten() #ravel 원본공유
print(ax.shape)
for i in range(10):
    # batch = it.next() #IDG에서 랜덤을 한번 작업 (변환) 
    batch = next(it)
    # print(batch.shape)
    image  = batch.reshape(100,100,3)
    ax[i].imshow(image)
    
    # row = i // 5
    # col = i % 5
    # if i < 5:  
    #     ax[0][i].imshow(batch)
    #     ax[0][i].axis('off')
    # else:
    #     ax[1][i-5].imshow(batch)
    #     ax[1][i-5].axis('off')
    ax[i].axis('off')
plt.tight_layout() #글자나 이미지가 서로 겹치지 않도록 레이아웃을 정리
plt.show()
