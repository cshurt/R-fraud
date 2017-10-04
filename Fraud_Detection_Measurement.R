# Necessary modules
# AUC
if (!require("AUC")) { 
  install.packages('AUC',
                   repos="https://cran.rstudio.com/",
                   quiet=TRUE)
  require('AUC') 
}

# Random Forest
if (!require("randomForest")) {
  install.packages('randomForest',
                   repos="https://cran.rstudio.com/",
                   quiet=TRUE)
  require('randomForest')
}

# Neural Nets
if (!require("nnet")) {
  install.packages('nnet',
                   repos="https://cran.rstudio.com/",
                   quiet=TRUE)
  require('nnet')
}

# Decision trees
if (!require("rpart")) {
  install.packages('rpart',
                   repos="https://cran.rstudio.com/",
                   quiet=TRUE)
  require('rpart')
}

# LASSO log likelihood
if (!require("glmnet")) { 
  install.packages('glmnet',
                   repos="https://cran.rstudio.com/",
                   quiet=TRUE)
  require('glmnet') 
}

# Tunemember Function
# tuneMember is found at this address: source("http://ballings.co/hidden/aCRM/code/chapter2/tuneMember.R")
# This function aids in tuning several models, using the AUC package
tuneMember <- function(call,tuning,xtest,ytest, predicttype=NULL,probability=TRUE){
  if (require(AUC)==FALSE) install.packages("AUC"); library(AUC)
  
  grid <- expand.grid(tuning)
  
  perf <- numeric()
  for (i in 1:nrow(grid)){
    Call <- c(as.list(call), grid[i,])
    model <-  eval(as.call(Call))
    
    predictions <- predict(model,xtest,type=predicttype, probability=probability)    
    
    if (class(model)[2] == "svm") predictions <- attr(predictions,"probabilities")[,"1"]
    
    
    if (is.matrix(predictions)) if (ncol(predictions) == 2 ) predictions <- predictions[,2]
    perf[i] <- AUC::auc(roc(predictions,ytest))
  }
  perf <- data.frame(grid, auc=perf)
  perf[which.max(perf$auc),]
}


# Read in the dataset
cardfraud <- read.csv('creditcard.csv')

# Data Exploration
head(cardfraud)
length(which(cardfraud$Class == 1))
nrow(cardfraud)
length(is.na(cardfraud))
sum(is.na(cardfraud))
cardfraud[which(cardfraud$Class == 1),]


# Dataset looks clean, let's divvy it up into three equally sized categories.

# Create indicators for non-fraud
normind <- sample(x=1:nrow(cardfraud[which(cardfraud$Class == 0),]),size =nrow(cardfraud[which(cardfraud$Class == 0),]))
trainindnor <- normind[1:round(length(normind)/3)]
valindnor <- normind[(round(length(normind)/3)+1):round(length(normind)*(2/3))]
testindnor <- normind[round(length(normind)*(2/3)+1):length(normind)]

# Create indicators for fraud
fraudind <- sample(x=1:nrow(cardfraud[which(cardfraud$Class == 1),]),size = nrow(cardfraud[which(cardfraud$Class == 1),]))
trainindfraud <- fraudind[1:round(length(fraudind)/3)]
valindfraud <- fraudind[(round(length(fraudind)/3)+1):round(length(fraudind)*(2/3))]
testindfraud <- fraudind[round(length(fraudind)*(2/3)+1):length(fraudind)]

# Make test, val, and train sets by combining fraud and non-fraud indicators
train <- rbind(cardfraud[which(cardfraud$Class == 0),][trainindnor,],cardfraud[which(cardfraud$Class == 1),][trainindfraud,])
val <- rbind(cardfraud[which(cardfraud$Class == 0),][valindnor,],cardfraud[which(cardfraud$Class == 1),][valindfraud,])
test <- rbind(cardfraud[which(cardfraud$Class == 0),][testindnor,],cardfraud[which(cardfraud$Class == 1),][testindfraud,])

# Set ytest, yval, and ytrain
ytrain <- as.factor(train$Class)
yval <- as.factor(val$Class)
ytest <- as.factor(test$Class)
ytrainbig <- as.factor(c(train$Class,val$Class))

# Drop the y values from the tables
train <- train[,!(names(train) %in% 'Class')]
val <- val[,!(names(val) %in% 'Class')]
test <- test[,!(names(test) %in% 'Class')]

# Make trainbig and ytrainbig
trainbig <- rbind(train,val)
ytrainbig <- as.factor(c(as.character(ytrain),as.character(yval)))

# Erase all variables except the necessary ones (saves on RAM)
rm(list=setdiff(ls(),c('train','val','test','ytrain','yval','ytest','trainbig','ytrainbig')))


## Decision Tree

# Grow a tree on trainbig
tree <- rpart(ytrainbig ~ .,
              trainbig,
              method = "class")

# Predict for all instances in test
predTree <- predict(tree,test)[,2]

# Compute the AUC
AUC::auc(roc(predTree,ytest))
AUC::auc(sensitivity(predTree,ytest))

# Make a nice plot of the tree
par(xpd = TRUE)
plot(tree, compress = TRUE)
text(tree, use.n = TRUE)

# Plot the AUROC
plot(roc(predTree,ytest))

## Random Forest
rFmodel <- randomForest(x=trainbig,
                        y=ytrainbig,
                        ntree=500,
                        importance=TRUE)
predrF <- predict(rFmodel,test,type='prob')[,2]

#assess model performance with AUROC
AUC::auc(roc(predrF,ytest))
# Assess model with AUSEC
AUC::auc(sensitivity(predrF,ytest))

# Plot the AUROC
plot(roc(predrF,ytest))

# A confusion matrix with a 0.5, 0.25, and 0.1 cutoff
table(ytest,predrF > 0.5)
table(ytest,predrF > 0.25)
table(ytest,predrF > 0.1)

## Zero Model
# An easy and foolish model
zeromod <- as.factor(numeric(94935))

# Accuracy of zero model
1-sum(as.numeric(as.character(ytest)))/nrow(test)

table(zeromod,ytest)

length(ytest)-sum(as.numeric(as.character(ytest)))

AUC::auc(roc(zeromod,ytest))


## Logistic Regression

LR <- glm(ytrainbig ~ .,
          data = trainbig,
          family = binomial("logit"))

(logisticsum <- summary(LR))
sort(abs(logisticsum$coefficients[,1]),decreasing = TRUE)

predLR <- predict(LR,
                  newdata=test,
                  type="response")


AUC::auc(roc(predLR,ytest))
AUC::auc(sensitivity(predLR,ytest))

# most important values
sort(abs(LR$coefficients))

## Neural Nets

# The data need to be scaled to train adequately
minima <- sapply(train,min)
scaling <- sapply(train,max)-minima

trainscale <- data.frame(base::scale(train,
                                    center=minima,
                                    scale=scaling))
sapply(trainscale,range)
valscale <- data.frame(base::scale(val,
                                   center=minima,
                                   scale=scaling))
testscale <- data.frame(base::scale(test,
                                    center=minima,
                                    scale=scaling))
trainbigscale <- data.frame(base::scale(trainbig,
                                        center=minima,
                                        scale=scaling))

# needs an NN.rang value and NN.maxit value
NN.rang <- 0.5
NN.maxit <- 10000
# Tuning values (can change depending on needs)
NN.size <- c(15,20,25,30)
NN.decay <- c(0.01,0.1)


# Create a call
call <- call("nnet",
             formula = ytrain ~ .,
             data = trainscale,
             rang = NN.rang,
             maxit = NN.maxit,
             trace = FALSE,
             MaxNWts = Inf)
tuning <- list(size=NN.size, decay=NN.decay)

# Call tuneMember
(result <- tuneMember(call=call,
                      tuning=tuning,
                      xtest=valscale,
                      ytest=yval,
                      predicttype='raw'))

# Test on ytrainbig
NN <- nnet(ytrainbig ~ .,
           trainbigscale,
           size = NN.size[4],
           rang = NN.rang,
           decay = NN.decay[3],
           maxit = NN.maxit,
           trace = TRUE,
           MaxNWtx= Inf)
predNN <- as.numeric(predict(NN,testscale,type='raw'))

AUC::auc(roc(predNN,ytest))
AUC::auc(sensitivity(predNN,ytest))

plot(roc(predNN,ytest))
plot(roc(predLR,ytest),add=TRUE, col = "blue")
plot(roc(predrF,ytest),add = TRUE, col = "red")
plot(roc(predTree,ytest),add=TRUE,col="green")
plot(roc(zeromod,ytest),add=TRUE,col="purple")

plot(sensitivity(predNN,ytest))
plot(sensitivity(predLR,ytest),add=TRUE, col = "blue")
plot(sensitivity(predrF,ytest),add = TRUE, col = "red")
plot(sensitivity(predTree,ytest),add=TRUE,col="green")
plot(sensitivity(zeromod,ytest),add=TRUE,col="purple")







# A function to clean things up and remove packages
detach_package <- function(pkg, character.only = FALSE)
{
  if(!character.only)
  {
    pkg <- deparse(substitute(pkg))
  }
  search_item <- paste("package", pkg, sep = ":")
  while(search_item %in% search())
  {
    detach(search_item, unload = TRUE, character.only = TRUE)
  }
}

detach_package(PRROC)
