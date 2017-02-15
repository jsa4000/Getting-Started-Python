import numpy as np


def L2(lambda_p):
    def wrap(params):
        return params + ' '+ str(lambda_p)
    return wrap

l2 = L2(0.002)
valueStr = l2('¿Qué he puesto?')

print (valueStr)

   
def L1(lambda_p = 0.001):
    def wrap(params):
        func = np.vectorize(lambda param: np.abs(param).sum())
        return (np.sum(func(params)) * lambda_p)
    return wrap

def L2(lambda_p = 0.0001):
    def wrap(params):
        func = np.vectorize(lambda param: np.power(param,2))
        return (np.sum(func(params)) * lambda_p)
    return wrap



W = [[1,2], [3,4]]
W2 = [[10,20] ,[30,40]]

myfunc = lambda param: np.abs(param).sum()

value = myfunc(W)
value2 = myfunc(W2)

print (value)
print (value2)

finl = value + value2

print(finl)


finl11 = L1()
finl12 = finl11([W,W2])


print (finl12)

regularizers = [L1(0.001),L2(0.0001)]

params = [W,W2]
print list(regularizer(params) for regularizer in regularizers)
params = [W,W2]
tot = np.sum(regularizer(params) for regularizer in regularizers)


print (tot)



