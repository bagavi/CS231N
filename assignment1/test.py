import numpy as np, time,random

y = [3,1,2]
def fun(x):
    return y[x]

x = map(fun,)
# 
# n = 10000
# classes = 10
# y = np.zeros(n)
# for i in range(n):
#     y[i] = random.randrange(0,classes)
# 
# x = np.zeros([n,classes])
# 
# tic = time.time()
# for i in range(n):
#     x[i][y[i]] = 1
# 
# print time.time() - tic
# 
# 
# z = [ [i,int(y[i])] for i in range(n)]
# print z
# x = np.zeros([n,classes])
# 
# tic = time.time()
# print x
# x[z] = 1
# print x
# print time.time() - tic
