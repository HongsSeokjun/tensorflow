import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

text = '오늘도 못생기고 영어를 디게 디게 디게 못 하는 이삭이는 재미없는 개그를 \
마구 마구 마구 마구 하면서 딴짓을 한다.'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
# {'디게': 1, '마구': 2, '오늘도': 3, '못생기고': 4, '영어를': 5, '못하는': 6, '이삭이는': 7, '재미없는': 8, '개그를': 9, '하면서': 10, '딴짓을': 11, '한다': 12}

print(token.word_counts)
# OrderedDict([('오늘도', 1), ('못생기고', 1), ('영어를', 1), ('디게', 3), ('못하는', 1), ('이삭이는', 1), ('재미없는', 1), ('개그를', 1), ('마구', 3), 
# ('하면서', 1), ('딴짓을', 1), ('한다', 1)])

x = token.texts_to_sequences([text])
print(x)
#[[3, 4, 5, 2, 2, 2, 6, 7, 8, 9, 10, 1, 1, 1, 1, 11, 12, 13]]

############## 원핫 3가지 맹그러봐 #############
# flattened = x[0]
#1. 판다스
# x = np.array(x[0])#.flatten() # (1,18) => (18,)
# x = pd.get_dummies(x)
# print(x.shape)
#2. sklearn
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
x= np.array(x[0])
x = x.reshape(-1, 1)
x = encoder.fit_transform(x)
print(x)
#3. keras
# from tensorflow.keras.utils import to_categorical
# x= np.array(x)#.flatten() 
# x = to_categorical(x)
# x = x[:,:,1:]
# x = x.reshape(18,13)
# print(x.shape) # 공칼람 문제 때문에 다르게 만들어줘야함