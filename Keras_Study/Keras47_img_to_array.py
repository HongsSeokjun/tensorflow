from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img # 이미지 땡겨오기.
from tensorflow.keras.preprocessing.image import img_to_array # 땡겨온 이미지 수치화
import numpy as np
import matplotlib.pylab as plt
from tensorflow.keras.models import load_model
path = 'C:\Study25\_data\image\me\\'

img = load_img(path+'me.jpg', target_size=(80,80),)
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
print(img.shape)
 
img = img /255
print(img)
np.save(path+'Keras47_me_catdog.npy',arr=img)