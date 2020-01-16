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

# Standardize variables
X = scale(X)
y = scale(y)

# GLMNET
library(glmnet, quietly = TRUE)
result = glmnet(X, y, alpha = 0.5, lambda = 10^seq(-2, 6, length.out = 5),
                 standardize = FALSE)

result.cv <- cv.glmnet(X, y, alpha = 0.5,
                       lambda = 10^seq(-2, 10, length.out = 50), nfolds = 10)
print(result.cv$lambda.min) # Best cross validated lambda

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
elastic_net_MM = function(y,X,alpha,lambda,beta_0=rep(0,p),epsilon=10^(-8)){
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
    
   # print(k)
    k = k + 1
    #print(l_old-l_new)
  }
  
  return(beta)
}

# p = length(X[1,]) # Number of parameters
# beta_0 = rep(0,p) # Initialize with beta_0 a vetor of 0s
# 
# alpha = 0.5
# lambda = 0.05428675
# elastic_net_MM(y,X,alpha,lambda,beta_0)


k_fold = function(y,X,k,alpha,lambda){
  # Function for k-fold crossvalidation, returns rmse for given alpha and lambda
n = length(y) # Number of observations
p = (dim(X)[2])
reshuffled_indices = sample(seq(1,n,1), n, replace=FALSE) # Shuffle indices
n_test = round(n/k) # if n=77 and k=10 we will have 8 obs. in the test set 
n_train = n-n_test
MSE=array(0,k-1)
for (i in seq(1,k-1,1)){
  # Divide data into test and training
  y_test = y[c(reshuffled_indices[(1+(i-1)*n_test):(i*n_test)])]
  y_train = y[-c(reshuffled_indices[(1+(i-1)*n_test):(i*n_test)])]
  X_test = X[c(reshuffled_indices[(1+(i-1)*n_test):(i*n_test)]),]
  X_train = X[-c(reshuffled_indices[(1+(i-1)*n_test):(i*n_test)]),]
  beta=elastic_net_MM(y_train,X_train,alpha,lambda,beta_0=rep(0,p)) ##get beta estimates from MM algorithm
  fitted_val=(X_test)%*%beta ##get y_hat
  MSE[i]=1/n_test*t(y_test-fitted_val)%*%(y_test-fitted_val) ##get MSE values
}
rmse=sqrt(sum(MSE)/(k-1))
return(rmse)
}

# Tunes hyperparameters alpha and lambda via k-fold crossvalidation. Returns optimal alpha and lambda
lambda_values = 10^seq(-2, 5, length.out = 50)
alpha_values = seq(0, 1, length.out = 10)
rmse=matrix(0,length(alpha_values),length(lambda_values)) ## create matrix, which will be filled with rmse values per hyperparamter combination
for (i in 1:length(alpha_values)){
  for (j in 1:length(lambda_values)){
    rmse[i,j]=k_fold(y,X,10,alpha_values[i],lambda_values[j]) ##Fill the matrix with rmse estimates
  }
}
plot(rmse[5,])
