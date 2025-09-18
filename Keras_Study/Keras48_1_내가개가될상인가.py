import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pylab as plt
from sklearn.metrics import accuracy_score
np_path = 'C:\Study25\_data\image\me\\'

x_pred= np.load(np_path+"Keras47_me_catdog.npy")

# scaler = MinMaxScaler() # 정규화
# x_pred1 = scaler.fit_transform(x_pred)
y_test = [1]
path = 'C:\Study25\_save\Keras44_catdog\\'

model = load_model(path+'k43_.hdf5')
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
print(y_predict)
images = x_pred.reshape(-1, 80, 80,3)


import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
if y_predict == 0:
    plt.title(f"예측:{'고양이'}")
else:
    plt.title(f"예측:{'강아지'}")
plt.imshow(images[0])
plt.show()

