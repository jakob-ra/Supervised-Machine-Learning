# TEST #
# Supervised Machine Learning - HW5 - #
# Maja Nordfeldt, Feb 2020 #

# -------------------------- Packages ---------------------------#

#if (!require("tidyverse")) install.packages("tidyverse")
if (!require("rpart")) install.packages("rpart")
if (!require("rpart.plot")) install.packages("rpart.plot")
if (!require("partykit")) install.packages("partykit")
#if (!require("kernlab")) install.packages("kernlab")

# -------------------------- Data Preparation --------------------#
data_url <- "github.com/jakob-ra/Supervised-Machine-Learning/blob/master/HW5/imdb.RData"
load((data_url))
#!!!! why / how set wd to git? to load data work

# Orig dataset
colnames(imdb)
sapply(imdb,class) # Checking classes of predictors
summary(is.na(df_imdb)) #NA's - quite many in gross_2015 budget, gross, budgeT_2015

# Selecting relevant subset
df_imdb <- imdb[!is.na(imdb$budget_2015),]
df_imdb <- df_imdb[!is.na(df_imdb$gross_2015),] # Remove obs with missing values
df_imdb <- df_imdb[(df_imdb[,15]>=1000000),]# Select movies only budget at least 1m

# Predict after production pre screening = drop post screening variables: imdb_score, gross, mpaa_rating, user_reviews critic_reviews, 
df_imdb <- df_imdb[,-c(7,8,9,11,13)]
# Assuming - no critcs viewed it, no users viewed it before, assuming actors credited will be known - ie on poster, 

# Making top actors separate variables
actorsplit <-strsplit(df_imdb$actors," \\| ")
df_actorsplit <- (matrix(unlist(actorsplit), ncol=3, byrow=TRUE,))
colnames(df_actorssplit) <- c("actor_1","actor_2","actor_3")
rbind(df_imdb,df_actorsplit)
test <- (unlist(actorsplit))
length(actorsplit)
dim(df_actorsplit)
dim(df_imdb)
summary(df_actorsplit)
# !!!! why unlist removes one obs? need to take each eleement of list and turn into row

# Make use of title: title length
df_imdb$title_length <- as.numeric(nchar(as.character(df_imdb$title)))
# Dummies for boolean var
df_imdb[,11:31] <- sapply(df_imdb[,11:31],as.numeric) 

X <- model.matrix( ~ factor(director) + factor(actor_1) + factor(actor_2) + factor(actor_3) + director_facebook_likes + cast_facebook_likes + budget_2015 + Drama + Comedy + Thriller + Action + Romance + Adventure + Crime + Fantasy + SciFi + Horror + Family +Mystery + Animation + Biography + Music + Sport + War + Musical + History + Documentary + Western + title_legth, data=df_imdb) # Including intercept, dropping one dummy for reference

df_imdb$gross_x3 <- as.factor(df_imdb$gross_2015 >= (df_imdb$budget_2015)*3) # 1 = movies gross box office takings of at least three times its budget 



# ------------------ Package results for reference --------------------#
setseed(1234)
test <- rpart(gross_x3 ~ director_facebook_likes + cast_facebook_likes + budget_2015 + Drama + Comedy + Thriller + Action + Romance + Adventure + Crime + Fantasy + SciFi + Horror + Family +Mystery + Animation + Biography + Music + Sport + War + Musical + History + Documentary + Western + title_legth, control = rpart.control(minsplit =10, minbucket = 5,xval=5,cp = 0.004,maxdepth = 10), data=df_imdb, method="class")
# !!!! Only use subset?
# !!!! specify again with all predictors included
# !!!! How to choose crossval?

plot(as.party(test))
test$splits
prp(test)

printcp(test) #display cp table
plotcp(test)# 	plot cross-validation results
rsq.rpart(test)	# plot approximate R-squared and relative error for different splits (2 plots). labels are only appropriate for the "anova" method.
print(test) #	print results
summary(test)#	detailed results including surrogate splits
plot(test)#	plot decision tree
text(test)	#label the decision tree plot
post(test, file=)#	create postscript plot of decision tre
importance(test)

# ------------------ Manual binary classification --------------------#
# Write your own R-function for a binary classification tree using the Gini index, based only on numeric features. Use the algorithm and splitting as described in the slides.

gini_index <- function(y_sorted){
  p_T <- summary(y_sorted)[2]/(summary(y_sorted)[1] + summary(y_sorted)[2]) #Fraction of TRUE
  p_F <- summary(y_sorted)[1]/(summary(y_sorted)[1] + summary(y_sorted)[2]) #Fractions of FALSE
  #print(p_T)
  #print(p_F)
  gini_index <- 1- (p_T^2) - (p_F^2)
  return(gini_index)
}

optimal_split_cont_gini_change <- function(data_left,cont_var,gini_parent){
  
  # Finding split points
  n_obs_to_split <- length(cont_var)
  cont_var<- sort(cont_var) # Arrange in increasing order 
  cont_var <- append(cont_var,0)
  next_element <- append(0,cont_var)
  split_candidates <- (cont_var + next_element)/2 # Splits between every two potential values 
  split_candidates <- split_candidates[-c(1,n_obs_to_split)] # Drop first and last split value
  
  # Storage for split comparison
  gini_parent <- rep(gini_parent,n_obs_to_split)
  wh_gini_change <- rep(NA,n_obs_to_split)
  
  # Calc improvement of splits
  for(i in i:length(split_candidates)){
    v_splitoutcomes <- (cont_var > split_candidates[i]) 
    bucket_above <- data_left[v_splitoutcomes,]
    bucket_below <- data_left[-v_splitoutcomes,]
    fraction_above <- (length(bucket_above)/n_obs_to_split)
    fraction_below <-(length(bucket_below)/n_obs_to_split)
    gini_above <- gini_index(bucket_above$outcome)
    gini_below <- gini_index(bucket_below$outcome)
    wh_gini_change[i] <- gini_parent - fraction_above*gini_above - fraction_below*gini_below
  }
  
  #Select best split
  m_results <- cbind(split_candidates,wh_gini_change)
  m_results <- m_results[order(m_results[,2]),] #Sort matrix with small to high by gini change
  best_split <- m_results[,dim(m_results)[2]]  # return split value and wh gini change
  
  return(best_split)

}

split_binary_gini_change <- function(data_to_sort,binary_var,gini_parent){
  n_obs_to_split <- length(binary_var)
  v_splitoutcomes <- (binary_var > 0.5) 
  bucket_1 <- data_to_sort[v_splitoutcomes,]
  bucket_0 <- data_to_sort[-v_splitoutcomes,]
  fraction_1 <- (length(bucket_1)/n_obs_to_split)
  fraction_0 <-(length(bucket_0)/n_obs_to_split)
  gini_1 <- gini_index(bucket_1[,34]) #obs sensitive to col n of outcome
  gini_0 <- gini_index(bucket_0[,34])#obs sensitive to col n of outcome
  wh_gini_change <- gini_parent - fraction_1*gini_1 - fraction_0*gini_0
  return(wh_gini_change)
}

super_splitter <- function(data_to_sort,variable_to_split,splitvalue){

  if(mean(variable_to_split)>0.5){ #need better choice condition
    v_splitoutcomes <- (variable_to_split > splitvalue) 
    node_1 <- data_to_sort[v_splitoutcomes,]
    node_2 <- data_to_sort[-v_splitoutcomes,]
     
  }
  else{
    v_splitoutcomes <- (variable_to_split > 0.5) 
    node_1 <- data_to_sort[v_splitoutcomes,]
    node_2 <- data_to_sort[-v_splitoutcomes,]
  }

  return()
  ## !!! not done - what to return
}

# Not done
tree_grower <- function(y,data_to_sort,min_obs,max_depth){
  
  data_to_sort <- data_to_sort
  k <- 0
  while(k < max_depth){
    k <- k+1
    1 <- no_of_splits
    no_nodes <- no_of_splits*2
    var_left_to_split <- dim(data_to_sort)[2]
    improve_of_var_left <- rep(NA,var_left_to_split,2)
    gini_parent <- gini_index(y) 
    
    for(i in 1:no_nodes){ #for all nodes in current level k
      
      if(length(data_to_sort) => min_obs){ #as long as enough obs left to sort
        
        
        
        for(i in 1:var_left_to_split){ #consider all possible variables to split (??)
          improve_of_var_left[i,1] <- i
          if(mean(data_to_sort[,i]) > 1){ 
            improve_of_var_left[i,2] <- optimal_split_cont_gini_change(data_to_sort,data_to_sort[,i],gini_parent)[2] 
            #store gini wh change of var
          }
          else{ #assuimg rest variables binary
            improve_of_var_left[i,2] <- split_binary_gini_change(data_to_sort,data_to_sort[,i],gini_parent)
            
          }
          #Eval which is the best var to split on 
          improve_of_var_left <- improve_of_var_left[order(improve_of_var_left[,2]),]
          select_to_split <- improve_of_var_left[1,var_left_to_split])
          print("Best var to choose is",improve_of_var_left[,var_left_to_split])
                  }
        
        
       # Execute best possible split by the splitting function
       # update node description, node assig vector and split table

      }
      else{ #move to next node
        
        
      }
       
    }
      
  }



# ------------------ Manual bagging of binary classification --------------------#
# Write also your own R-function which applies bagging to binary classification trees built using rpart()

