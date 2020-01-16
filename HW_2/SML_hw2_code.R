# *------------------------------------------------------------------
# | PROGRAM NAME: Supervised Machine Learning - HW 2
# | DATE: 16-1-2020
# | CREATED BY: Jakob Rauch
# *----------------------------------------------------------------

# Clear cache
rm(list = ls())

# Load supermarket data from github
githubURL = "https://github.com/jakob-ra/Supervised-Machine-Learning/raw/master/HW_2/supermarket1996.RData"
load(url(githubURL))
df = subset(supermarket1996, select = -c(STORE, CITY, ZIP, GROCCOUP_sum, SHPINDX) )
attach(df)

# create summary statistics
summary(supermarket1996) 

# Correlation matrix
cor(df)

# Create vector y (turnover)
y = GROCERY_sum

# Create matrix X of predictor variables
X = subset(df, select = -GROCERY_sum)

# Standardize predictor variables
X = scale(X)

# GLMNET
library(glmnet, quietly = TRUE)
result = glmnet(X, y, alpha = 1, lambda = 10^seq(-2, 6, length.out = 5),
                 standardize = FALSE)

library(plotrix)
plot(result, xvar = "lambda", label = TRUE, las = 1)
legend("bottomright", lwd = 1, col = 1:6, bg = "white",
          legend = pasteCols(t(cbind(1:ncol(X), " ",colnames(X)))), cex = .7)


# Loss function
loss = function(y,X,alpha,lambda,beta,epsilon){
  # Loss function given the data, parameters beta, hyperparameters lambda and alpha, 
  # and convergence threshold epsilon. Returns both the value of the loss function and the D matrix
  n = length(y) # Number of observations
  p = length(beta) # Number of parameters
  D = matrix(0,p,p) # Initialize p times p matrix of 0s
  beta_abs_list = rep(0,p)
  for (j in seq(0,p,1)){
    beta_abs_list[j] = abs(beta[j])
    D[j,j] = 1/max(c(beta_abs_list[j],epsilon))
  }
  c = 1/(2*n)*t(y)%*%y + 1/2*lambda*alpha*sum(beta_abs_list)
  l = 1/2*t(beta)%*%(1/n*t(X)%*%X + lambda*(1-alpha)*diag(p) + lambda*alpha*D)%*%beta - 1/n*t(beta)%*%t(X)%*%y + c

  return(list(l,D))
}

# MM-algorithm
elastic_net_MM = function(y,X,alpha,lambda,beta_0,epsilon=10^(-8)){
  # Fits an elastic net model via MM-algorithm. Returns vector beta_hat 
  # for given data ,hyperparameters lambda and alpha, intitial parameter guess beta_0, and convergence threshold epsilon
  n = length(y) # Number of observations
  p = length(beta_0) # Number of parameters
  
  l_new = 0 # so that (l_old-l_new)/l_old > epsilon can be evaluated as true
  l_old = 1 # so that (l_old-l_new)/l_old > epsilon can be evaluated as true
  beta = beta_0
  k = 1
  while ((l_old-l_new)/l_old > epsilon){
    l_old = loss(y,X,alpha,lambda,beta,epsilon)[[1]]  # Gets value l of loss function (FUTURE: try not to call function twice)
    D = loss(y,X,alpha,lambda,beta,epsilon)[[2]] # Gets matrix d, which we already computed in the loss function
    A = 1/n*t(X)%*%X + lambda*(1-alpha)*diag(p) + lambda*alpha*D
    beta = solve(A, 1/n*t(X)%*%y)
    l_new = loss(y,X,alpha,lambda,beta,epsilon)[[1]]
    
    print(k)
    k = k + 1
    print(l_old-l_new)
  }
  
  return(beta)
}

p = length(X[1,]) # Number of parameters
beta_0 = rep(0,p) # Initialize with beta_0 a vetor of 0s

alpha = 0.5
lambda = 0.0000001
elastic_net_MM(y,X,alpha,lambda,beta_0)

# epsilon = 10^(-8)
# beta = beta_0
# dim(loss(y,X,alpha,lambda,beta,epsilon)[[2]])
# dim(X)
