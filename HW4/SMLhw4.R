set.seed(123456)

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
    qtilde*((qtilde<=1)&(y==1)) + ((qtilde>1)&(y==1))
  # A = diag(1,nrow(X)) # skip for computational simplicity
  vnew = solve(t(X)%*%X + lambdaP)%*%t(X)%*%b
  
  return(vnew)
}

## SVMMaj MM algorithm

SVMMaj = function(y,X,lambda,v0,epsilon = 1e-5){
  # returns optimal v = (c,w)' given data y, X, penalty lambda
  # by running the MM algorithm with initial level v0 and stopping criterion
  # epsilon
  
  lambdaP = diag(lambda,ncol(X))
  lambdaP[1,1] = 0
  
  Lold = loss.svm.quad(y,q = X%*%v0,w = v0[-1],lambda)
  cat('L0 =',Lold,"\n")
  k = 1
  Lnew = Lold
  v = v0
  while(k==1|((Lold-Lnew)/Lold)>epsilon) {
    k = k+1
    Lold = Lnew
    vnew = v.update(y,X,v,lambdaP)
    Lnew = loss.svm.quad(y,q = X%*%vnew,w = vnew[-1],lambda)
    v = vnew
    cat('after iteration ',k," L = ",Lnew,"\n")
  }
  cat("stopping crit achieved\n")
  return(vnew)
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
vhat = SVMMaj(y,X,lambda,v0,epsilon = 1e-5)
vhat






