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

# Create vector y (turnover)
y = GROCERY_sum

# Create matrix X of predictor variables
X = subset(df, select = -GROCERY_sum)

# Standardize variables
X = scale(X)
y = scale(y)

# Loss function for MM-algorithm
loss = function(y,X,alpha,lambda,beta,epsilon){
  # Loss function of MM-algorithm given the data, parameters beta, hyperparameters lambda and alpha, 
  # and convergence threshold epsilon. Returns both the value of the loss function and the D matrix
  
  n = length(y) # Number of observations
  p = length(beta) # Number of parameters
  
  D = matrix(0,p,p) # Initialize p times p matrix of 0s
  beta_abs_list = rep(0,p) # Vector to hold absolute values of beta (will need them later)
  
  for (j in seq(0,p,1)){
    beta_abs_list[j] = abs(beta[j])
    D[j,j] = 1/max(c(beta_abs_list[j],epsilon))
  }
  
  c = 1/(2*n)*t(y)%*%y + 1/2*lambda*alpha*sum(beta_abs_list)
  l = 1/2*t(beta)%*%(1/n*t(X)%*%X + lambda*(1-alpha)*diag(p) + lambda*alpha*D)%*%beta - 1/n*t(beta)%*%t(X)%*%y + c

  return(list(l,D))
}

# MM-algorithm for minimizing the elastic net loss function
elastic_net_MM = function(y,X,alpha,lambda,beta_0=rep(0,(dim(X)[2])),epsilon=10^(-8)){
  # Fits an elastic net model via MM-algorithm. Returns vector beta_hat for given data,
  # hyperparameters lambda and alpha, intitial parameter guess beta_0 (default is a vector of 0s)
  # and convergence threshold epsilon (default is 10^(-8)).
  
  n = length(y) # Number of observations
  p = (dim(X)[2]) # Number of parameters
  
  # Set values for the first iteration
  beta = beta_0
  k = 1
  # Set l_new and l_old so that (l_old-l_new)/l_old > epsilon is true in the first iteration
  l_new = 0 
  l_old = 1 
  
  # Iterate the approximation until convergence
  while ((l_old-l_new)/l_old > epsilon){
    l_and_D = loss(y,X,alpha,lambda,beta,epsilon) # Loss returns both the value of the loss function and D
    l_old = l_and_D[[1]]  # Gets value l of loss function 
    D = l_and_D[[2]] # Gets matrix D, which we already computed in the loss function
    A = 1/n*t(X)%*%X + lambda*(1-alpha)*diag(p) + lambda*alpha*D
    beta = solve(A, 1/n*t(X)%*%y)
    l_new = loss(y,X,alpha,lambda,beta,epsilon)[[1]]
    
   # print(k)
    k = k + 1
    #print(l_old-l_new)
  }
  
  return(beta)
}


k_fold_crossval = function(y,X,k,alpha,lambda){
  # Function for k-fold crossvalidation, returns RMSE for given alpha and lambda
  n = length(y) # Number of observations
  p = (dim(X)[2]) # Number of parameters
  
  # Reshuffle data and split into k groups of equal size
  reshuffled_indices = sample(seq(1,n,1), n, replace=FALSE) # Shuffle indices of our n observations
  n_test = floor(n/k) # n=77 and k=10 would give 7 obs. in the test set the rest of the observations
  # in the training set. For n=77, using k=11 or k=7 is advisable to get evenly sized groups.

  MSE=array(0,k) # Vector to hold the MSEs for each fold
  
  # Loop over the k folds and save MSEs
  for (i in seq(1,k,1)){
    # Divide data into test and training
    y_test = y[c(reshuffled_indices[(1+(i-1)*n_test):(i*n_test)])]
    y_train = y[-c(reshuffled_indices[(1+(i-1)*n_test):(i*n_test)])]
    X_test = X[c(reshuffled_indices[(1+(i-1)*n_test):(i*n_test)]),]
    X_train = X[-c(reshuffled_indices[(1+(i-1)*n_test):(i*n_test)]),]
    
    beta = elastic_net_MM(y_train,X_train,alpha,lambda) # get beta estimates for training data from MM algorithm 
    fitted_val = (X_test)%*%beta # get y_hat for test data
    MSE[i]=1/n_test*t(y_test-fitted_val)%*%(y_test-fitted_val) # save MSE values for test data
  }
  
  RMSE = sqrt(sum(MSE)/k)
  
  return(RMSE)
}


min_RMSE = function(y,X,k=7,alpha_values=seq(0,1,length.out=3),lambda_values=10^seq(-3, 4, length.out = 10)){
  # Tunes hyperparameters alpha and lambda via k-fold crossvalidation. Returns optimal combination of alpha and lambda
  # as well as the resulting RMSE.
  
  RMSE = matrix(0,length(alpha_values),length(lambda_values)) # create matrix to hold RMSE for each hyperparamter combination

  # Loop over hyperparameter combinations
  for (i in 1:length(alpha_values)){
    for (j in 1:length(lambda_values)){
      RMSE[i,j] = k_fold_crossval(y,X,k,alpha_values[i],lambda_values[j]) # Fill the matrix with RMSE
    }
  }
  
  min_index = arrayInd(which.min(RMSE), dim(RMSE)) # Find index of lowest RMSE
  alpha_min = alpha_values[min_index[1]] # alpha value at lowest RMSE
  lambda_min = lambda_values[min_index[2]] # lambda value at lowest RMSE
  RMSE_min = RMSE[min_index] # lowest RMSE
  
  return(list(alpha_min,lambda_min,RMSE_min))
}

  
## We want to compare our result with glmnet function, which cannot cross-validate alpha -> compare for fixed alpha
alpha = 0.5

# Find best cross validated lambda from glmnet, letting the function choose the sequence of lambdas to try
result.cv <- cv.glmnet(X, y, alpha=alpha, folds=7)
print(result.cv$lambda.min) # Note that the result here varies quite a bit but is frequently equal or close to 0.05

# Find best cross validated lambda from our own function, with a fine sequence of lambdas between 0 and 1
print(min_RMSE(y,X,alpha_values=alpha,lambda_values=seq(0.01,1,0.01))[[2]]) # Results are equal or very close


## Knowing that everything works as intended, find the best combination of alpha and lambda
result = min_RMSE(y,X,alpha_values=seq(0,1,0.1),lambda_values=seq(0.01,1,0.01)) # This takes a while
print(result)

# Get point estimates on full sample for the optimal values of alpha and lambda
beta_hat = elastic_net_MM(y,X,0.6,0.04)

# set very low point estimates (rounding errors) to 0
zero_ind = arrayInd(which(beta_hat < 10^(-5)), dim(beta_hat)) 
beta_hat[zero_ind] = 0
print(beta_hat)
