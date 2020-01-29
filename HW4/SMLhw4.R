set.seed(123456)

### functions  

## calculate q g
q =function(X,c,w) c+X%*%w

## SVM loss
loss.svm = function(y,q,w,lambda){
  # returns SVM loss given weights, predicted qs
  grp0err = sum((q+1)[(y==-1)&(q>-1)])
  grp1err = sum((1-q)[(y==1)&(q<1)])
  return(grp0err+grp1err+lambda*sum(w^2))
} 



## quadratic hinge -- notation as in slides
hinge.quad = function(y,q,qtilde){
  # returns sum of individual quadratic hinges
  b = qtilde[qtilde<=-1&y==-1] - (q>-1&y==-1) + 
    qtilde[qtilde<=1&y==1] + (qtilde>1&y==1)
  
  c = qtilde[qtilde<=-1&y==-1] + (q>-1&y==-1) + 
     (qtilde<=1&y==1) + (2-qtilde+(qtilde-1)^2)(qtilde>1&y==1)
  
  return(sum(q^2 -2*b*q +c))
}

## SVMMaj MM algorithm



### main

# load data
load('./bank.RData')
smp = sample(1:dim(bank)[1],size = 1000,replace = F)
bank = bank[smp,]
y = 2*as.numeric(bank$y)-3   # scale to -1,1 instead of 1,2

# transform categorical variables to dummies
# omit euribor3m, weird entries.
X = model.matrix(~.-euribor3m -1 ,data = bank)
X = scale(X)
               

# linear SVM
k = ncol(X)
w=rep(0,k)
loss.svm(y,q=q(X,c=1,w=w),w ,lambda=0)

