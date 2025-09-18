import numpy as np

a = np.array(range(1, 11))

timesteps = 5

print(a.shape) # (10,)

# def split_x(x, y):
#     aaa = []
#     for i in range(len(x)- y +1):
#         subset = x[i : (i+y)]
#         aaa.append(subset)
#     return np.array(aaa)

def split_xy(dataset, timesteps):
    x, y = [], []
    for i in range(len(dataset) - timesteps+1):
        x_window = dataset[i : i + timesteps-1]
        y_label = dataset[i + timesteps-1]
        x.append(x_window)
        y.append(y_label)
    return np.array(x), np.array(y)


# bbb = split_x(a, timesteps)
# print(bbb)
x,y = split_xy(a,timesteps)
# x = bbb[:,:-1]
# y = bbb[:,-1]
print(x,y)
print(x.shape,y.shape)

