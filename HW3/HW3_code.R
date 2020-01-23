# *------------------------------------------------------------------
# | PROGRAM NAME: Supervised Machine Learning - HW 3
# | DATE: 23-1-2020
# | CREATED BY: Jakob Rauch
# *----------------------------------------------------------------

# Clear cache
rm(list = ls())

# Load supermarket data from github
githubURL = "https://github.com/jakob-ra/Supervised-Machine-Learning/raw/master/HW3/Airline.RData"
load(url(githubURL))
attach(Airline)

# y
y = output
y = scale(y)

# Get dummies
X = model.matrix(output ~ -1 + factor(airline) + factor(year) + cost + pf + lf + factor(airline)*cost + factor(airline)*pf + factor(airline)*lf + factor(year)*cost + factor(year)*pf + factor(year)*lf + cost*pf + cost*lf + pf*lf + cost^2 + pf^2 + lf^2 + factor(airline)*cost^2 + factor(airline)*pf^2 + factor(airline)*lf^2 + factor(year)*cost^2 + factor(year)*pf^2 + factor(year)*lf^2 + (cost*pf)^2 + (cost*lf)^2 + (pf*lf)^2)

X = scale(X)
# X = c(X[,1:20],scale(X[,-(1:20)]))

K_RBF = function(X,gamma){
  # Returns kernel matrix K for a radial basis function with parameter gamma
  n = dim(X)[1]

  K = matrix(NA,n,n)
  
  for (i in 1:n){
    x_i = X[i,]
    
    for (j in 1:n){
      x_j = X[j,]
      x_diff = x_i-x_j
      x_diff_sum = sum(x_diff^2)
      k_ij = exp(-gamma*x_diff_sum)
      K[i,j] = k_ij
    }
  }
  
  return(K)
}


K_IHP = function(X,d){
  # Returns kernel matrix K for a inhomogeneous polynomial kernel for degree d (integer >= 1)
  n = dim(X)[1]
  
  K = matrix(NA,n,n)
  
  for (i in 1:n){
    x_i = X[i,]
    
    for (j in 1:n){
      x_j = X[j,]
      x_ij_dot = t(x_i) %*% x_j
      k_ij = (1+x_ij_dot)^d
      K[i,j] = k_ij
    }
  }
  
  return(K)
}
 

q_tilde = function(y,K_inv,lambda){
  # Returns vector q_tilde for a given K_inv (from Kernel) and parameter gamma
  return(solve(diag(dim(K_inv)[1]) + lambda*K_inv,y))
}


# Compare manual to package for RBF
manual = q_tilde(y,solve(K_RBF(X,1/2)),0.5)

res = krr(y, X, lambda=0.5, kernel.type = "RBF", kernel.RBF.sigma = 1/dim(X)[2])
package = res$yhat

print(cbind(manual,package))

# Compare manual to package for IHP
manual = q_tilde(y,solve(K_IHP(X,2)),0.5)

res = krr(y, X, lambda=0.5, kernel.type = "nonhompolynom", kernel.degree = 2)
package = res$yhat

print(cbind(manual,package))

# rmse = cv.krr(y[1:80], X[1:80,], lambda=0.5, kernel.type = "nonhompolynom", yu = y[81:90], Xu = X[81:90,], kernel.degree = 2)
