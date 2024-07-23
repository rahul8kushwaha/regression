import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
class LinearRegressor:
    """
    This class implements LinearRegression from scratch
        fit(X:input in numpy, Y:output in numpy): function for getting the linear regression coefficients
        predict(X:input in numpy): function for inference on input
    """
    def __init__(self, alpha = 0.001, eplison_2 = 0.001):
        self.alpha = alpha
        self.eplison_2 = eplison_2
        self.J_theta = 100
    def fit(self, X, Y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        n = X.shape[1]
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0)+1e-5
        self.x_std[0] = 1
        self.x_mean[0] = 0
        X = np.subtract(X, self.x_mean)
        X = np.divide(X, self.x_std)
        theta = np.random.rand(n).reshape(-1,1)
        epoch = 1
        B = max((X.shape[0]//16), 16)
        while self.J_theta >= self.eplison_2:
            loss = 0
            for batch in range(math.ceil(X.shape[0]/B)):
                self.J_theta = ((np.matmul(X[B*batch:B*batch+B,:], theta)-Y[B*batch:B*batch+B,:])**2).mean()
                grad_J_theta = np.matmul(X[B*batch:B*batch+B,:].T, (np.matmul(X[B*batch:B*batch+B,:], theta)-Y[B*batch:B*batch+B,:]))
                theta = theta - self.alpha*grad_J_theta
                loss += self.J_theta
            print(f"epoch: {epoch} \t loss: {loss}")
            epoch += 1
        self.theta = theta
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        X = np.subtract(X, self.x_mean)
        X = np.divide(X, self.x_std)
        return np.matmul(X, self.theta)
