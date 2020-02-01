set.seed(123456)

library(SVMMaj)

### functions  

## calculate q g
# q =function(X,c,w) c+X%*%w

## SVM loss
loss.svm.quad= function(y,q,w,lambda){
  # returns SVM loss given weights, predicted qs
  grp0err = sum(((q+1)[(y==-1)&(q>-1)])^2)
  grp1err = sum(((1-q)[(y==1)&(q<1)])^2)
  return(grp0err+grp1err+lambda*sum(w^2))
} 



## quadratic hinge -- notation as in slides
v.update = function(y,X,vold,lambdaP){
  # returns updated v+
  qtilde = X%*%vold
  b = qtilde*((qtilde<=-1)&(y==-1)) - ((qtilde>-1)&(y==-1)) + 
    ((qtilde<=1)&(y==1)) + qtilde*((qtilde>1)&(y==1))
  # A = diag(1,nrow(X)) # skip for computational simplicity
  vnew = solve(t(X)%*%X + lambdaP)%*%t(X)%*%b
  
  return(vnew)
}

## SVMMaj MM algorithm

SVMMaj_manual = function(y,X,lambda,v0,epsilon = 1e-5){
  # returns optimal v = (c,w)' given data y, X, penalty lambda
  # by running the MM algorithm with initial level v0 and stopping criterion
  # epsilon
  
  lambdaP = diag(lambda,ncol(X))
  lambdaP[1,1] = 0
  
  Lold = loss.svm.quad(y,q = X%*%v0,w = v0[-1],lambda)
  #cat('L0 =',Lold,"\n")
  k = 1
  Lnew = Lold
  v = v0
  while(k==1|((Lold-Lnew)/Lold)>epsilon) {
    k = k+1
    Lold = Lnew
    vnew = v.update(y,X,v,lambdaP)
    Lnew = loss.svm.quad(y,q = X%*%vnew,w = vnew[-1],lambda)
    v = vnew
    #cat('after iteration ',k," L = ",Lnew,"\n")
  }
  #cat("stopping crit achieved\n")
  return(vnew)
}
### k-fold
k_fold_crossval = function(y,X,k,lambda,v0){
  # Function for k-fold crossvalidation, returns RMSE for given alpha and lambda
  n = length(y) # Number of observations
  p = (dim(X)[2]) # Number of parameters
  
  # Reshuffle data and split into k groups of equal size
  reshuffled_indices = sample(seq(1,n,1), n, replace=FALSE) # Shuffle indices of our n observations
  n_test = floor(n/k) # n=77 and k=10 would give 7 obs. in the test set the rest of the observations
  # in the training set. For n=77, using k=11 or k=7 is advisable to get evenly sized groups.
  
  loss=array(0,k) # Vector to hold the loss for each fold
  
  # Loop over the k folds and save MSEs
  for (i in seq(1,k,1)){
    # Divide data into test and training
    y_test = y[c(reshuffled_indices[(1+(i-1)*n_test):(i*n_test)])]
    y_train = y[-c(reshuffled_indices[(1+(i-1)*n_test):(i*n_test)])]
    X_test = X[c(reshuffled_indices[(1+(i-1)*n_test):(i*n_test)]),]
    X_train = X[-c(reshuffled_indices[(1+(i-1)*n_test):(i*n_test)]),]
    
    beta = SVMMaj_manual(y_train,X_train,lambda,v0) # get beta estimates for training data from majorization algorithm 
    fitted_val = (X_test)%*%beta # get qhat for test data
    TP=sum(fitted_val>1&y_test==1)
    TN=sum(fitted_val<(-1)&y_test==(-1))
    loss[i]=1-(TP+TN)/n_test # save missclassification error values for test data
  }
  loss=mean(loss) ##take average over found losses
  
  return(loss)
}

min_loss = function(y,X,k=10,lambda_values=10^seq(-3, 4, length.out = 10),v0){
  # Tunes hyperparameters lambda via k-fold crossvalidation. Returns optimal lambda
  # as well as the resulting loss
  
  loss = array(0,length(lambda_values)) # create vector to hold loss for each lambda
  
  # Loop over hyperparameter combinations
  for (j in 1:length(lambda_values)){
    loss[j] = k_fold_crossval(y,X,k,lambda_values[j],v0) # Fill the matrix with losses
  }
  
  
  min_index = which.min(loss) # Find index of lowest loss
  lambda_min = lambda_values[min_index] # lambda value at lowest loss
  loss_min = loss[min_index] # lowest misclassification error
  
  return(list(loss,lambda_min,loss_min))
}
### main

# load data
load('./bank.RData')
smp = sample(1:dim(bank)[1],size = 1000,replace = F)
bank = bank[smp,]
y = 2*as.numeric(bank$y)-3   # scale to -1,1 instead of 1,2

# transform categorical variables to dummies
# omit euribor3m, weird entries.
X = model.matrix(y~.-euribor3m -emp.var.rate -1 ,data = bank)
# no observations in education illiterate, but model matrix command creates dummy with NANs for it:

X = scale(X)
idx.missing = which(colSums(is.na(X))>0) # identify index of column containing NAns
X = X[,-idx.missing]
X = cbind(1,X)               

# linear SVM
p = ncol(X)-1

# magic numbers
lambda = 0.01
v0 = rep(0,p+1)

# lets go
vhat = SVMMaj_manual(y,X,lambda,v0,epsilon = 1e-5)
print(vhat)

#mis_error=k_fold_crossval(y,X,10,lambda=0.1,v0)
man_results=min_loss(y,X,10,lambda_values=seq(0.01, 1, length.out = 20),v0=v0) ### find optimal lambda
loss_vector=man_results[[1]]
loss_vector
opt_lambda=man_results[[2]]
plot(loss_vector,type="l")
vhat=SVMMaj_manual(y,X,opt_lambda,v0,epsilon = 1e-5) ##our solution with optimal lambda


##Compare with package

pack_results=svmmajcrossval(X,y,ngroup=10,search.grid = list(lambda=seq(0.01, 1, length.out = 20))) ##Find optimal lambda
cbind(opt_lambda,pack_results$param.opt)
pack_vhat=svmmaj(X,y,opt_lambda,hinge="quadratic",scale="none") ###get coefficients for optimal beta
cbind(vhat,pack_vhat$beta) ## compare our solution to package
