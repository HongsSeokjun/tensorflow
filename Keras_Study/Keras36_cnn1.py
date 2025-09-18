from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D # 이미지라 2D


# 원본은 N,5,5,1 이미지
model = Sequential()
model.add(Conv2D(10,(2,2),input_shape=(5,5,1)))#10output,(2,2)커널사이즈 => 가중치
# (None, 4, 4, 10) defalt로 (4,4,1) 인식  50
model.add(Conv2D(5,(2,2)))
# (None, 3, 3, 5)     205
model.summary()
#파라미터 수=(커널 높이×커널 너비×입력 채널 수+1)×출력 채널 수