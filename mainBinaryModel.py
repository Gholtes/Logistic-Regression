import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

class logit:
    """Fits a binary logit model with MLE"""
    def __init__(self, x, y,
                        B=None,
                        n=1,
                        v=False,
                        maxIterations = 500,
                        minGrad = 0.00001,
                        method = "NR"):
        '''Agruments:
        x: a column wise array of explanatory variables
        y: Target values that take values of 1 or 0
        B: initial values for the coefficients, if desired
        n: Learning rate
        method: How to fit the model:
            'NR' = Newton Raphson
            'GD'=Gradient descent
        v: Verbose, prints gradients and results
        maxIterations: Max number of iterations in the fitting process
        minGrad: minimum gradient that triggers fitting process to stop
        '''
        ###PARAM
        self.exp_corr = 0.000001

        #Add in the constant term
        const_ = np.ones(x.shape[0])
        x = np.c_[const_, x]
        self.x = x

        #init coeff
        if B == None:
            B = np.random.rand(1, x.shape[1])
        else:
            B = np.transpose(np.array([B]))
            print(B)
            print(B.shape)
            if B.shape != (1, x.shape[1]):
                print("ERROR:Shape of B incorrect, replacing with random values of correct shape")
                B = np.random.rand(1, x.shape[1])

        #define the logit, S(XB) function
        def S(x, B): #CHECKED
            '''Returns a column vector Nx1'''
            XB = np.matmul(B, np.transpose(x))
            denom = np.add(1, np.exp(np.negative(XB)))
            return np.transpose(np.add(np.divide(1, denom), self.exp_corr))

        def dlogLik_dB(x, y, B):
            '''Returns a row vector 1xK+1 for each B'''
            y_SXB = np.add(y, np.negative(S(x, B)))
            return np.matmul(np.transpose(y_SXB), x)

        def d2logLik_dB2(x, y, B):
            '''Returns a row vector 1xK+1 for each B'''
            XtX = np.square(x)
            _1_S = np.add(1, np.negative(S(x, B)))
            S_1_S = np.multiply(S(x, B), _1_S)
            return np.transpose(np.matmul(np.transpose(XtX), S_1_S))

        def score(x, y, B):
            '''Returns a row vector 1xK+1 for each B'''
            return np.transpose(np.matmul(np.transpose(x), np.add(y, np.negative(S(x, B)))))

        def Hessian(x, y, B):
            _1_S = np.add(1, np.negative(S(x, B)))
            S_1_S = np.multiply(S(x, B), _1_S)
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

        def learn(x, y, B, v, maxIterations, minGrad):
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
                if np.abs(np.mean(dLL)) <= minGrad or np.abs(np.mean(d2LL)) == 0 and v:
                    if v:
                        print("-- Minimum gradient reached --")
                    break
                else:
                    B = updateB(B, dLL, d2LL, n)
                    if (iter-1) % 10 == 0 and v:
                        print("Iteration: {0}".format(iter))
                        print("    Fitted Values: {0}".format(B))
                        print("    Gradient:      {0}".format(dLL))
                        print("____________________________________")
            if v:
                print("Number of Newton-Raphson iterations: {0}".format(iter))
                print("Fitted Values: {0}".format(B))
                print("____________________________________")
            self.iter = iter
            return B
        #Assign results
        self.coefficients = learn(x, y, B, v, maxIterations, minGrad)
        self.fitted = S(x, B)


    def __repr__(self):
        return "Number of Newton-Raphson iterations: {0} \nFitted Values: {1}".format(self.iter, self.coefficients)

    def predict(self, x=[]):
        if len(x) == 0:
            x = self.x
        else:
            #add ones and reshape
            x = np.array(x)
            if len(x.shape) == 1:
                x = np.transpose(np.array([x]))
            elif len(x.shape) < 1:
                print("ERROR: x dimensions greater than 2. Please use a column matrix of dim 2")

            #Try force the input data into shape
            if x.shape[1] != self.x.shape[1]-1:
                x = np.transpose(x)
                if x.shape[1] != self.x.shape[1]-1:
                    print("ERROR: x columns don't match training data columns")
                    print("Please use a column matrix of dim 2, where each variable is a column")

            const_ = np.ones(x.shape[0])
            x = np.c_[const_, x]

        B = self.coefficients
        def S(x, B): #CHECKED
            '''Returns a column vector Nx1'''
            XB = np.matmul(B, np.transpose(x))
            denom = np.add(1, np.exp(np.negative(XB)))
            return np.transpose(np.add(np.divide(1, denom), self.exp_corr))
            x
        return S(x, B)

filename = "data.csv"
dt = pd.read_csv(filename)

#Make data into np arrays
y = (np.array(dt[["y"]]))
x = np.array(dt[["x1", "x2", "x3"]])

#Fit the model by calling the logit class with data
model = logit(x, y)

#Show model summary
print(model)

#Extract B values
B = model.coefficients

#method 1 to get fitted values of y
y_ = model.fitted
#method 2 to get fitted values, can accept new data as model.predict(x=testData)
y_ = model.predict()

x1 = dt["x1"]
x2 = dt["x2"]
x3 = dt["x3"]

#Plot fitted values by axis
plt.plot(x1,y, ".")
plt.plot(x1,y_, "b.")
plt.plot(x3,y_, "r.")
plt.plot(x2,y_, "g.")

plt.show()
