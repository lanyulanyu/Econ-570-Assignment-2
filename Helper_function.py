import pandas as pd
import numpy as np
import random
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
np.random.seed(10)

### Define a function to generate variance-covariance matrix
def fn_generate_cov(dim, corr):
    #Inputs
    # -dim: number of covariates

    acc  = []
    for i in range(dim):
        row = np.ones((1,dim)) * corr
        row[0][i] = 1
        acc.append(row)
    return np.concatenate(acc,axis=0)



### Define a function to generate correlated random variables
def fn_generate_multnorm(nobs,corr,nvar):
    #Inputs
    # -nobs: number of observations
    # -corr: correlation for multivariate normal
    # -nvar: number of covariates generated

    mu = np.zeros(nvar)
    std = (np.abs(np.random.normal(loc = 1, scale = .5,size = (nvar,1))))**(1/2)
    # generate random normal distribution
    acc = []
    for i in range(nvar):
        acc.append(np.reshape(np.random.normal(mu[i],std[i],nobs),(nobs,-1)))

    normvars = np.concatenate(acc,axis=1)

    cov = fn_generate_cov(nvar, corr)
    C = np.linalg.cholesky(cov)

    Y = np.transpose(np.dot(C,np.transpose(normvars)))
    return Y


### Define a function to randomly separate the samples to treated and non treated
def fn_randomize_treatment(N,p=0.5):
    #Inputs
    # -N: Number of observations (Sample size)
    # -p: the proportion of treated people in the sample

    treated = random.sample(range(N), round(N*p))
    return np.array([(1 if i in treated else 0) for i in range(N)]).reshape([N,1])



### Define a function to calculate the bias, rmse and size of treatment parameter
def fn_bias_rmse_size(theta0,thetahat,se_thetahat,cval = 1.96):
    #Inputs
    # -theta0: true parameter value
    # -thetatahat: estimated parameter value
    # -se_thetahat: estimated standard error of thetahat

    b = thetahat - theta0
    bias = np.mean(b)
    rmse = np.sqrt(np.mean(b**2))
    tval = b/se_thetahat # paramhat/se_paramhat H0: b = 0
    size = np.mean(1*(np.abs(tval)>cval))

    return (bias,rmse,size)


