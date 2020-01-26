
load_airlinedata = function(s){
  load(s)
  y = Airline[[4]]
  X = model.matrix(~factor(Airline$airline)+factor(Airline$year)+
                     as.matrix(Airline[,c(3,5,6)])-1)
  X = scale(X)
  print(dim(X))
  # X = model.matrix(~scale(Airline$cost)) # work with  p = 1 for visualizing
  y = scale(y)
  return(list(y,X))
}


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
  n = dim(X)[1] 
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
    RMSFE.gauss = RMSFE.gauss + 1/k*sum((ytest-qnew)^2)
    
    # inhom. polynomial kernel: 
    Kall = K_IHP(X,d)
    Ktrain = Kall[train,train]
    Ku = Kall[!train,train]
    qnew  = mean(ytrain) + Ku%*%est.qtilde.gauss(ytrain-mean(ytrain),Ktrain,lambda)
    # qall[!train] = qnew
    RMSFE.poly = RMSFE.poly + 1/k*sum((ytest-qnew)^2)
    
  }
  
  # # plot
  #  mycol = rainbow(k)
  #  idx = order(X[,2])
  #  plot(X[,2],y)
  #  points(X[idx,2],qall[idx],pch=16,col='blue')
  
  
  # cat("RMSFE for lambda =",lambda,': ',RMSFE)
  return(c(RMSFE.gauss,RMSFE.poly))
}



cvalkrr.lambda = function(y,X,fold,gamma,d,lambdagrid){
  # returns RMSFE for a grid of lambdas by k fold cross validation
  
  RMSFE = matrix(NA,length(lambdagrid),2) # initialize grid of average RMSFE for each lambda
  
  for (l in 1:length(lambdagrid)) RMSFE[l,] = av.rmsfe(y,X,fold,gamma,d,lambdagrid[l])
  
  return(RMSFE)
}

## gaussian kernel 
phi.gaussian = function(X,gamma) exp(-gamma*as.matrix(dist(X)^2)) 

## other kernel  
K_IHP = function(X,d) (1+X%*%t(X))^d

main = function(){
  s = 'Airline.RData'
  dta = load_airlinedata(s)
  y = dta[[1]]
  X = as.matrix(dta[[2]])
  k = 9 
  
  lambdagrid = seq(10^(-4), 20, length.out = 100)
  gamma = 1/23
  d=2
  
  
  
  # assign data randomly to one of the k bins 
  n = length(y)
  ntest = floor(n/k) # observations per bin
  rank = rank(rnorm(n))
  fold = rep(1:k,each = ntest) 
  fold = fold[rank] # vector of assigned bin
  
  RMSFE = cvalkrr.lambda(y,X,fold,gamma,d,lambdagrid) # matrix of RMSFE for different lambda
  
  
  plot(lambdagrid,RMSFE[,1],type='l',main = 'Gaussian kernel')
  plot(lambdagrid,RMSFE[,2],type='l',main = 'Inhom. polynomial kernel') 
  
  # print(colMins(RMSFE))
  
  # # plot
  # idx = order(X[,2])
  # plot(X[,2],y,main="Kernel Ridge regression, gaussian kernel, whole sample")
  # lines(X[,2][idx],qtilde.gauss[idx],col='blue')
}


main()




