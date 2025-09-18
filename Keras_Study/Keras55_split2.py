import numpy as np

a = np.array([[1,2,3,4,5,6,7,8,9,10],
              [9,8,7,6,5,4,3,2,1,0]])
timesteps = 6
def split_xy(dataset, timesteps):
    x, y = [], []
    for i in range(len(dataset) - timesteps+1):
        x_window = dataset[i : i + timesteps-1]
        y_label = dataset[i + timesteps-1][1]
        x.append(x_window)
        y.append(y_label)
    return np.array(x), np.array(y)

x,y = split_xy(a.transpose(),timesteps)

print(x)
print(x.shape)
print(y)
print(y.shape)
# 10행 2열의 데이터중에
# 첫번째와 두번째 칼럼을 x로 잡고,
# 두번째 칼럼을 y값으로 잡는다.

#(6,4,2),=> y값 (6,1)