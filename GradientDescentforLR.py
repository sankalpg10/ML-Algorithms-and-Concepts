# Gradient Descent for Linear Regression
# yhat = b1x + b0 
# loss = (y-(b1x+b0))**2 / N 
import numpy as np
# Initialise some parameters
x = np.random.randn(10,1)
y = 5*x + np.random.rand()
# Parameters
b1 = 0.0 
b0 = 0.0 
# Hyperparameter 
learning_rate = 0.01


def gradient_descent(x,y,b1,b0,learning_rate):
    N = x.shape[0]

    dldb1 = 0.0
    dldb0 = 0.0

    for xi,yi in zip(x,y):
        
        dldb1 += -2*xi*(yi - (b1*xi +b0))

        dldb0 += -2*(yi - (b1*xi +b0))

    #update coefficients

    b1 = b1 - learning_rate*(1/N)*(dldb1)
    b0 = b0 - learning_rate*(1/N)*(dldb0)

    return b1,b0

N = x.shape[0]
for epoch in range(250):

    b1,b0 = gradient_descent(x,y,b1,b0,0.01)

    yhat = b1*x + b0

    loss = np.divide(np.sum((y-yhat)**2, axis=0), x.shape[0]) 
    print(f'{epoch} loss is {loss}, paramters b1:{b1}, b0:{b0}')

print(x,"\n",y,"\n",yhat)