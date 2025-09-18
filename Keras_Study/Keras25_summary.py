from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#2. 모델
model = Sequential([
    Dense(3, input_dim=1),
    Dense(2),
    Dense(4),
    Dense(1)
])
model.summary()