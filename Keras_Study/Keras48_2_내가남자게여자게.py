import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pylab as plt
from sklearn.metrics import accuracy_score
np_path = 'C:\Study25\_data\image\me\\'

x_pred= np.load(np_path+"Keras47_me.npy")

# scaler = MinMaxScaler() # 정규화
# x_pred1 = scaler.fit_transform(x_pred)
y_test = [0]
path = 'C:\Study25\_save\Keras46_gender\\'
model = load_model(path+'k46_8.hdf5')
#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])

#4. 평가, 예측
print('#############################')

#_predict = model.predict(x_test)
#y_predict =  (y_predict > 0.5).astype(int)
y_predict = model.predict(x_pred)
y_predict =  (y_predict > 0.5).astype(int)
accuracy_score = accuracy_score(y_test, y_predict)
print('accuracy_score :', accuracy_score)
print(y_predict.shape)

images = x_pred.reshape(-1, 250, 250,3)

images = np.clip(images, 0, 255).astype(np.uint8) 
# import matplotlib
# matplotlib.rcParams['font.family'] = 'Malgun Gothic'
# if y_predict == 0:
#     plt.title(f"예측:{'남자'}")
# else:
#     plt.title(f"예측:{'여자'}")
# plt.imshow(images[0])
# plt.show()

import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(4, 4))
for i in range(1):
    plt.imshow(images[i])
    if y_predict[0][0] == 0:
        plt.title(f"예측:{'남자'}")
    else:
        plt.title(f"예측:{'여자'}")
    plt.axis('off')
plt.tight_layout()
plt.show()

