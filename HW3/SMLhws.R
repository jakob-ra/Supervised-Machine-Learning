# *------------------------------------------------------------------
# | PROGRAM NAME: Supervised Machine Learning - HW 3
# | DATE: 23-1-2020
# | CREATED BY: Jakob Rauch
# *----------------------------------------------------------------

# Clear cache
rm(list = ls())

# Import libraries
library(dsmle)

est.qtilde.gauss = function(y,K,lambda){
  # input: nx1 vector y (mean zero), nxn kernel matrix (K), 
  #         tuning parameter lambda
  # return: Kinvq, nx1 vector of in sample predictions minimizing kernel rifge regression loss
  #         multiplied by K^-1
  
  n = dim(K)[1]
  Kinvq = solve(K+lambda*diag(n))%*%y
 
  return(Kinvq)
}


av.rmsfe = function(y,X,fold,gamma,d,lambda){
  # returns average RMSFE for a given lambda over k bins, for both kernels
  n = dim(X)[1] # nr of observations
  k = max(fold) # nr of bins
  
  # qall.gauss = rep(NA,n) # store out of sample predictions for plot
  RMSFE.gauss = 0
  RMSFE.poly = 0
  
  for (i in 1:k){
    # split the data
    train = (fold!=i)  # indicator training data
    ytrain = y[train]
    ytest = y[!train]
    
    # gaussian kernel: 
    Kall = phi.gaussian(X,gamma)
    Ktrain = Kall[train,train]
    Ku = Kall[!train,train]
    qnew  = mean(ytrain) + Ku%*%est.qtilde.gauss(ytrain-mean(ytrain),Ktrain,lambda)
    # qall[!train] = qnew
    RMSFE.gauss = RMSFE.gauss + 1/k*mean((ytest-qnew)^2)
    
    # inhom. polynomial kernel: 
    Kall = K_IHP(X,d)
    Ktrain = Kall[train,train]
    Ku = Kall[!train,train]
    qnew  = mean(ytrain) + Ku%*%est.qtilde.gauss(ytrain-mean(ytrain),Ktrain,lambda)
    # qall[!train] = qnew
    RMSFE.poly = RMSFE.poly + 1/k*mean((ytest-qnew)^2)
    
  }
  
  # # plot
  #  mycol = rainbow(k)
  #  idx = order(X[,2])
  #  plot(X[,2],y)
  #  points(X[idx,2],qall[idx],pch=16,col='blue')
  
  
  # cat("RMSFE for lambda =",lambda,': ',RMSFE)
  return(sqrt(c(RMSFE.gauss,RMSFE.poly)))
}



cvalkrr.lambda = function(y,X,fold,gamma,d,lambdagrid){
  # returns RMSFE for a grid of lambdas by k fold cross validation
  
  RMSFE = matrix(NA,length(lambdagrid),2) # initialize grid of average RMSFE for each lambda
  
  for (l in 1:length(lambdagrid)) RMSFE[l,] = av.rmsfe(y,X,fold,gamma,d,lambdagrid[l])
  
  return(RMSFE)
}

## gaussian kernel 
phi.gaussian = function(X,gamma) exp(-gamma*as.matrix(dist(X)^2)) 

## inhomogeneous polynomial kernel  
K_IHP = function(X,d) (1+X%*%t(X))^d


# Load supermarket data from github
githubURL = "https://github.com/jakob-ra/Supervised-Machine-Learning/raw/master/HW3/Airline.RData"
load(url(githubURL))
attach(Airline)

y = output
X = model.matrix(~ -1 + factor(airline) + year + cost + pf + lf)[,-1] # gets the X matrix with 5 airline dummies

# Rescale
X = scale(X)
y = scale(y)

p = dim(X)[2] # Nr of predictors
k = 9 # Nr of folds

lambdagrid = 10^seq(-8, 0, length.out = 100) # values to try for lambda
gamma = 1/p # fix gamma at 1/p
d=2 # fix d at 2



# assign data randomly to one of the k bins 
n = length(y)
ntest = floor(n/k) # observations per bin
rank = rank(rnorm(n))
fold = rep(1:k,each = ntest) 
fold = fold[rank] # vector of assigned bin

RMSFE = cvalkrr.lambda(y,X,fold,gamma,d,lambdagrid) # matrix of RMSFE for different lambda

min_index = which.min(RMSFE[,1]) # Find index of lowest RMSE
lambda_min_gaussian = lambdagrid[min_index] # lambda value at lowest RMSE

min_index = which.min(RMSFE[,2]) # Find index of lowest RMSE
lambda_min_poly = lambdagrid[min_index] # lambda value at lowest RMSE

# Plot lambda vs RMSE
op = par(mfrow = c(1, 2))
plot(lambdagrid,RMSFE[,1],type='l', log='x', main = 'Gaussian kernel',
     xlab="Lambda", ylab="RMSE")
abline(v=lambda_min_gaussian, col="purple")
plot(lambdagrid,RMSFE[,2],type='l', log='x', main = 'Inhom. polynomial kernel',  xlab="Lambda", ylab="RMSE") 
abline(v=lambda_min_poly, col="purple")
par(op)



# Compare manual to package for RBF
ker.cv.rbf = cv.krr(as.vector(y), X, k.folds = 9, lambda = lambdagrid,
                center = F, scale = F, kernel.type = "RBF", kernel.RBF.sigma = 1/2*p)
ker.cv.poly = cv.krr(as.vector(y), X, k.folds = 9, lambda =  lambdagrid,
                    center = F, scale = F, kernel.type = "nonhompolynom", kernel.degree = 2)


op = par(mfrow = c(1, 2))
plot(ker.cv.rbf, ylim = c(0, 2), las = 1)
plot(ker.cv.poly, ylim = c(0, 2), las = 1)
par(op)


# Compare optimal lambda, manual vs package
cat('The optimal lambda for the gaussian kernel is', 
    lambda_min_gaussian,
    'according to our manual function and', ker.cv.rbf$lambda.min, 'according to the cv.krr function. The resulting RMSE are',
    min(RMSFE[,1]), 'and',  min(ker.cv.rbf$rmse), '\n')



cat('The optimal lambda for inhomogeneous polynomial kernels is', lambda_min_poly,
    'according to our manual function and', ker.cv.poly$lambda.min, 'according to the cv.krr function. The resulting RMSE are',
    min(RMSFE[,2]), 'and',  min(ker.cv.poly$rmse), '\n')
