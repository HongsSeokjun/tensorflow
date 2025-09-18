import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255, # 0~255 스케일링, 정규화
    horizontal_flip=True, # 수평 뒤집기 <- 데이터 증폭 또는 변환
    vertical_flip= True, # 수직 뒤집기 <- 데이터 증폭 또는 변환
    width_shift_range=0.1, # 평행이동 10%
    height_shift_range=0.1, # 수직이동 10%
    rotation_range=5, # 회전 5도
    zoom_range=1.2, # 줌 1.2배
    shear_range=0.7, # 좌표하나를 고정시키고, 다른 몇개의 좌표를 이동시키는 변환(찌부 만들기)
    fill_mode='nearest',
)

test_datagen = ImageDataGenerator(
    rescale=1./255, # 0~255 스케일링, 정규화
) # 평가 데이터는 변환하지 않는다.

path_train = 'C:\Study25\_data\image\\brain\\train\\'
path_test ='C:\Study25\_data\image\\brain\\test\\'

# 파일을 열꺼야
xy_train = train_datagen.flow_from_directory(
    path_train,             # 경로
    target_size=(200,200),  # 리사이즈, 사이즈 규격 일치, 큰거는 축소, 작은거는 확대
    batch_size=10,          # 160개의 사진을 (200,200,1) 16번으로 나누는 이유는 메모리 문제!
    class_mode='binary',    # 
    color_mode='grayscale',
    shuffle=True,
    seed=333,
) # Found 160 images belonging to 2 classes. //2 classes = ad, normal

xy_test = test_datagen.flow_from_directory(
    path_test,             # 경로
    target_size=(200,200),  # 리사이즈, 사이즈 규격 일치, 큰거는 축소, 작은거는 확대
    batch_size=10,          # 120개의 사진을 (200,200,1) 12번으로
    class_mode='binary',    # 
    color_mode='grayscale',
    #shuffle=True,
) # Found 120 images belonging to 2 classes.
# 튜플과 리스트 수정 유무 + => 소괄호 대괄호 차이
print(xy_train) #<keras.preprocessing.image.DirectoryIterator object at 0x000001A0B0B742B0> => 먼저 batch로 분리한 데이터로 넣는거
#*이터레이터(Iterator)**는 데이터의 연속적인 요소에 순차적으로 접근할 수 있도록 해주는 객체
print(xy_train[0]) 
# x, y를 각각 데이터 np형태로 10개씩 담겨져 있음
# batch 10개의 ad, normal => 0과 1로 [1., 0., 1., 0., 1., 0., 1., 1., 0., 0.]
#print(len(xy_train)) #16 번
# print(xy_train[0][0])  #x 데이터(10, 200, 200, 1)
#print(xy_train[0][1])  #y 데이터(10,)
 
#print(xy_train[0].shape) AttributeError: 'tuple' object has no attribute 'shape'
#print(xy_train[16].shape) Asked to retrieve element 16, but the Sequence has length 16
#print(xy_train[0][2]) IndexError: tuple index out of range

# print(type(xy_train)) #<class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0])) #<class 'tuple'>
# print(type(xy_train[0][0])) #<class 'numpy.ndarray'> -> 0번째 배치의 x 데이터
# print(type(xy_train[0][1])) #<class 'numpy.ndarray'> -> 0번째 배치의 y 데이터