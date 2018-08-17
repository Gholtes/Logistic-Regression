import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

###PARAM
n = 1
exp_corr = 0.000001
#read
filename = "data.csv"
dt = pd.read_csv(filename)

#Make data into np arrays
y = (np.array(dt[["y"]]))
# x = np.array(dt[["x1", "x2", "x3"]])
x = np.array(dt[["x1"]])

const_ = np.ones(x.shape[0]) #Add intercept
x = np.c_[const_, x]
#init coeff
B = np.random.rand(1, x.shape[1])
#define the logit function
def logit(x, B): #CHECKED
    '''Returns a column vector Nx1'''
    XB = np.matmul(B, np.transpose(x))
    denom = np.add(1, np.exp(np.negative(XB)))
    return np.transpose(np.add(np.divide(1, denom), exp_corr)) #TODO

def dlogLik_dB(x, y, B):
    '''Returns a row vector 1xK+1 for each B'''
    y_SXB = np.add(y, np.negative(logit(x, B)))
    return np.matmul(np.transpose(y_SXB), x)

def d2logLik_dB2(x, y, B):
    '''Returns a row vector 1xK+1 for each B'''
    XtX = np.square(x)
    _1_S = np.add(1, np.negative(logit(x, B)))
    S_1_S = np.multiply(logit(x, B), _1_S)
    return np.transpose(np.matmul(np.transpose(XtX), S_1_S))

def score(x, y, B):
    '''Returns a row vector 1xK+1 for each B'''
    return np.transpose(np.matmul(np.transpose(x), np.add(y, np.negative(logit(x, B)))))

def Hessian(x, y, B):
    _1_S = np.add(1, np.negative(logit(x, B)))
    S_1_S = np.multiply(logit(x, B), _1_S)
    diag_S_1_S =np.diag(np.squeeze(np.transpose(S_1_S)))
    Xt_S1S = np.matmul(np.negative(np.transpose(x)), diag_S_1_S)
    return np.matmul(Xt_S1S, x)

def updateB(B, dLL, d2LL, n=n, method = "NR"):
    if method == "NR":
        ''' ~~Newton Raphson method~~'''
        dll_d2LL = np.multiply(n, np.divide(dLL, d2LL))
        return np.add(B, dll_d2LL)
    elif method == "GD":
        ''' ~~Gradient descent method~~'''
        return np.add(B, np.multiply(n, dLL))
    #SCORE METHOD
    # dll_d2LL = np.matmul(score(x, y, B), np.linalg.inv(Hessian(x, y, B)))
    # return np.add(B, dll_d2LL)

def learn(x, y, B, maxIterations = 500, min_grad = 0.00001):
    '''
    ~~Newton Raphson method~~
    is iteration > maxIterations?
        if yes, quit and return B and error
    get dLL, d2LL
    check if dervive is above threshold
        if yes update B
        if not quit and return B
    '''
    iter = 0 #EDGE - avoids maxIterations = 0 from causing error
    for iter in range(maxIterations):
        dLL = dlogLik_dB(x, y, B)
        d2LL = d2logLik_dB2(x, y, B)
        if np.abs(np.mean(dLL)) <= min_grad or np.abs(np.mean(d2LL)) == 0:
            print("-- Minimum gradient reached --")
            break
        else:
            B = updateB(B, dLL, d2LL, n)
            if (iter-1) % 10 == 0:
                print("Iteration: {0}".format(iter))
                print("    Fitted Values: {0}".format(B))
                print("    Gradient:      {0}".format(dLL))
                print("____________________________________")
    print("Number of Newton-Raphson iterations: {0}".format(iter))
    print("Fitted Values: {0}".format(B))
    print("____________________________________")
    return B

B = learn(x, y, B)
y_ = logit(x, B)
x1 = dt["x1"]
x2 = dt["x2"]
x3 = dt["x3"]

plt.plot(x1,y, ".")
plt.plot(x1,y_, "b.")
# plt.plot(x3,y_, "r.")
# plt.plot(x2,y_, "g.")

plt.show()
