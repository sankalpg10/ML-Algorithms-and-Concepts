from math import sqrt
from numpy import asarray
from numpy import arange
from numpy.random import rand
from numpy.random import seed
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# objective function
def objective(x, y):
    return x**2.0 + y**2.0

# derivative of objective function
def gradients(x, y):
 return asarray([x * 2.0, y * 2.0])

def adam(objective,gradients,bounds,n_iters,beta1,beta2,alpha,eps=1e-8):
    """_summary_

    Args:
        params (_type_): _description_
        gradients (_type_): gradients of the params
        beta1 (_type_): Hyperparameter for first moment(mean)
        beta2 (_type_): Hyperparameter for second moment(uncentered variance)
        alpha (_type_): Hyperparameter for Stepsize
        eps (_type_): Very small value to avoid division by zero
    """

    solutions = list()
    #set an initial point
    x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    # print(f"x: {x}")

    score = objective(x[0], x[1])

    # initialize first and second moments
    m = [0.0 for _ in range(bounds.shape[0])]
    # print(f"m: {m}")
    v = [0.0 for _ in range(bounds.shape[0])]
    # print(f"v: {v}")

    # run the gradient descent updates

    for t in range(n_iters):

        #calculate gradient - g(t)

        grads = gradients(x[0],x[1])
        print(f'grads : {grads}')

        # build a solution one variable at a time
        print(f"bounds.shape[0]: {bounds.shape[0]}")
        for i in range(bounds.shape[0]):

            #update first moment

            m[i] = beta1*m[i] + (1-beta1)*grads[i]

            #updates second moment

            v[i] = beta2*v[i] + (1-beta2)*grads[i]**2

            #correct moment bias

            m_hat = m[i]/(1 - beta1**(t+1))
            v_hat = v[i]/(1 - beta2**(t+1))

            x[i] = x[i] - alpha*m_hat/(sqrt(v_hat) + eps)

        # evaluate candidate point
        score = objective(x[0], x[1])
        # keep track of solutions
        solutions.append(x.copy())
        # report progress
        print('>%d f(%s) = %.5f' % (t, x, score))
        
    return solutions


        

# seed the pseudo random number generator
seed(1)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 60
# steps size
alpha = 0.04
# factor for average gradient
beta1 = 0.83
# factor for average squared gradient
beta2 = 0.999
# perform the gradient descent search with adam
solutions = adam(objective, gradients, bounds, n_iter, beta1, beta2,alpha)
# sample input range uniformly at 0.1 increments
xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
yaxis = arange(bounds[1,0], bounds[1,1], 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)
# create a filled contour plot with 50 levels and jet color scheme
pyplot.contourf(x, y, results, levels=50, cmap='jet')
# plot the sample as black circles
solutions = asarray(solutions)
pyplot.plot(solutions[:, 0], solutions[:, 1], '.-', color='w')
# show the plot
pyplot.show()