# TEST #
# Supervised Machine Learning - HW5 - #
# Maja Nordfeldt, Feb 2020 #

# -------------------------- Packages ---------------------------#

# rm(list = ls())
library(tidyverse)
library(partykit)
library(randomForest)
library("rpart")
library("rpart.plot")

# -------------------------- Data Preparation --------------------#

githubURL = "https://github.com/jakob-ra/Supervised-Machine-Learning/raw/master/HW5/imdb.RData"
load(url(githubURL))
#!!!! why / how set wd to git? to load data work

# Orig dataset
colnames(imdb)
sapply(imdb,class) # Checking classes of predictors


# Selecting relevant subset
df_imdb <- imdb[!is.na(imdb$budget_2015),] # drop missing budget
df_imdb <- df_imdb[!is.na(df_imdb$mpaa_rating),] # drop missing MPAA
df_imdb <- df_imdb[!is.na(df_imdb$gross_2015),] # Remove obs with missing values
df_imdb <- df_imdb[(df_imdb[,15]>=1000000),]# Select movies only budget at least 1m

# Make use of title: title length
df_imdb$title_length <- as.numeric(nchar(as.character(df_imdb$title)))
titles = df_imdb$title

# drop post screen variables
dropvars = c('imdb_score','gross', 'budget','user_reviews','critic_reviews','director','actors','title','year')
dropind = colnames(df_imdb)%in%dropvars
df_imdb <- df_imdb[,!dropind] # dropping

# create factor out of genre dummies


# outcome
df_imdb$gross_x3 <- as.factor(df_imdb$gross_2015 >= (df_imdb$budget_2015)*3) # 1 = movies gross box office takings of at least three times its budget 

# Plot budget vs revenue, line with slope 3
plot(df_imdb$budget_2015,df_imdb$gross_2015)
abline(a=0,b=3)

plot(log(df_imdb$budget_2015),log(df_imdb$gross_2015),
     ylab = 'log gross',xlab='log budget')
xgrid = seq(0.1,4e8,length.out = 1000)
lines(log(xgrid),log(3*xgrid),col='red',lwd=2)
legend('bottomright',legend = 'gross = 3*budget',col='red',lwd = 2)

df_imdb = subset(df_imdb, select = -gross_2015) # drop gross asa  predictor

# Missing values
summary(is.na(df_imdb))

# Analyse data imbalance
summary(df_imdb$gross_x3)

# ------------------ Package results for reference --------------------#

# small tree for visual explanation

treesmall = rpart(gross_x3 ~ . - gross_x3 -mpaa_rating, 
                  control = rpart.control(minsplit =5, minbucket = 5,xval=5,cp = 0.004,maxdepth = 3), 
                  data=df_imdb, method="class")

plot(as.party(treesmall))


# Single tree - cp = 1e-7 to get nice cp plot
set.seed(1)
tree = rpart(gross_x3 ~ . - gross_x3, 
             control = rpart.control(minsplit =5, minbucket = 5,xval=10,cp = 1e-7,maxdepth = 10), 
             data=df_imdb, method="class")
plotcp(tree)# 	plot cross-validation results

cpprint = printcp(tree)
cvmin = cpprint[which.min(cpprint[,4]),1]
cat('best cp:',cvmin,'\n')
cat('min xerror =:',cpprint[which.min(cpprint[,4]),4],'\n')


# Single tree - cp = 0.006 --> cross validated parameter
set.seed(1)
tree = rpart(gross_x3 ~ . - gross_x3, 
             control = rpart.control(minsplit =5, minbucket = 5,xval=5,cp = cvmin,maxdepth = 10), 
             data=df_imdb, method="class")

pred = predict(tree, type="class")
tree.conf = table(df_imdb$gross_x3,pred)
tree.conf.error = c(tree.conf[1,2]/(tree.conf[1,1]+tree.conf[1,2]),tree.conf[2,1]/(tree.conf[2,1]+tree.conf[2,2]))
tree.conf = cbind(tree.conf,tree.conf.error) # attach confusion error
colnames(tree.conf) = c('pred 0','pred 1' ,'class.error')
rownames(tree.conf) = c('obs 0','obs 1' )

knitr::kable(tree.conf)


# plot the tree
plot(as.party(tree))
tree$splits
prp(tree)
 
# ------ Bagging --------
set.seed(1)
bag = randomForest(gross_x3 ~ . - gross_x3, data = df_imdb, mtry = length(df_imdb)-1, ntree = 500)

# results
colnames(bag$confusion) = c('pred 0','pred 1' ,'class.error')
rownames(bag$confusion) = c('obs 0','obs 1' )
knitr::kable(bag$confusion)

# plot(bag, lwd = 2)
varImpPlot(bag)
plot(bag,lwd=2,main = 'Classification error')
legend('right',legend = c('Class 0','Class 1','overall'),col=c('red','green','black'),lty= c(2,3,1))

# ------------------ Manual binary classification --------------------
# Write your own R-function for a binary classification tree using the Gini index, based only on numeric features. Use the algorithm and splitting as described in the slides.

