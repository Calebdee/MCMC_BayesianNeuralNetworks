import numpy as np
import copy
import matplotlib.pyplot as plt


def transition(mu, variance):
    return np.random.normal(mu, variance, 1)

def log_joint(z):
    # Get log joint
    return -z*z + np.log(affine_sigmoid(z, m=10, c=3) + 1e-6)

def to_accept(x, x_new):
    if x_new>x:
        return True
    else:
        accept=np.random.uniform(0,1)
        return (accept < (np.exp(x_new-x)))

def prior(x):
    if(x[1] <=0):
        return 0
    return 1


def affine_sigmoid(xin, m= 10, c= 3):
    if type(xin) != np.ndarray:
        x = np.array([xin])
    else:
        x = xin

    x = get_affine(x, m, c)
    output = get_sigmoid(x)

    if type(xin) != np.ndarray:
        return output[0]
    else:
        return output

def get_affine(x, m, c):
    x = m*x + c
    return x

def get_sigmoid(x):
    output = np.zeros(x.shape)
    ind1 = (x >= 0)
    ind2 = (x  < 0)
    output[ind1] = 1 / (1 + np.exp(-x[ind1]))
    output[ind2] = np.divide(np.exp(x[ind2]), (1 + np.exp(x[ind2])))

    return output

def gass_hermite_quad(f, degree, m, c):
    points, weights = np.polynomial.hermite.hermgauss( degree)
    f_x = f(points, m= m, c= c)
    F = np.sum( f_x  * weights)

    return F

def log_joint_gradient(z):
    return -2*z + (1-affine_sigmoid(z, m=10, c=3))*10

def log_joint_gradient2(z, mean, sigma):
    data_minus_mean = z[np.newaxis,:] - mean
    sigma_inv       = np.linalg.inv(sigma) 
    gradient        = - np.matmul(sigma_inv, data_minus_mean.T)
    gradient        = gradient[0]
    return gradient


def accept_hybrid(deltaH):
    if (deltaH <= 0.0):
        return True
    else:
        u = np.random.uniform(0.0,1.0)
        if (u < np.exp(-deltaH)):
            return True
        else:
            return False

def accept_log(x, x_new):
    if x_new > x:
        return True
    else:
        accept = np.random.uniform(0,1)
        return (accept < (x_new)/(x+1e-5))

def Uzt(phi, pred, t, dot_product, weight, reg= 1):
    prior = -0.5* np.sum(np.multiply(weight, weight))
    likelihood = np.multiply(t, np.log(pred+1e-5)) + np.multiply(1.0- t, np.log(1.0-pred+1e-5))
    likelihood = np.sum(likelihood)

    return prior + likelihood

def Krt(p):
    p = p.flatten()
    return 0.5*np.sum(np.multiply(p, p))

def sigmoid(weights, data):
    dot_product = np.matmul(data, weights)
    output = np.zeros(dot_product.shape)
    ind1 = (dot_product >= 0)
    ind2 = (dot_product  < 0)
    output[ind1] = 1 / (1 + np.exp(-dot_product[ind1]))
    output[ind2] = np.divide(np.exp(dot_product[ind2]), (1 + np.exp(dot_product[ind2])))

    return output, dot_product

def accuracy(pred, labels):
    if pred.ndim == 2:
        pred = pred[:,0]
    pred[pred >= 0.5] = 1.0
    pred[pred <  0.5] = 0.0
    acc = np.sum(pred == labels)*100.0/pred.shape[0]
    return acc

def prediction_likelihood(data, labels, weight):
    pred, _ = sigmoid(weight, data)
    pred = pred[:,0]
    pred_like = np.multiply(labels, np.log(pred + 1e-10)) + np.multiply(1.0-labels, np.log(1.0-pred+ 1e-10))
    return np.exp(np.mean(pred_like))


z = np.arange(-3, 3+0.01, 0.01)
N = gass_hermite_quad(affine_sigmoid, degree= 100, m= 10, c= 3)
pz = np.multiply(np.exp(-np.multiply(z, z)), affine_sigmoid(z))/N