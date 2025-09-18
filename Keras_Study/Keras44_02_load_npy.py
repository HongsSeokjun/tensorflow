import numpy as np
import time

np_path = 'C:\Study25\_data\_save_npy\\'
#np.save(np_path+'Keras44_01_x_train.npy', arr=x)
#np.save(np_path+'Keras44_01_y_train.npy', arr=y)

start = time.time()
x_train = np.load(np_path+"Keras44_01_x_train.npy")
y_train = np.load(np_path+"Keras44_01_y_train.npy")
end = time.time()

print(x_train.shape, y_train.shape) #(25000, 200, 200, 3) (25000,)
print(x_train)
print(y_train[:20])
print("시간 :",round(end-start,3))
#시간 : 29.278