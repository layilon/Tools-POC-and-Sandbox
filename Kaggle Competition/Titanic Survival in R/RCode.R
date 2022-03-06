library(dplyr)
library(tidyverse)
library(readxl)
library(ggplot2)
library(glmnet)
library(mice)
library(MASS)
library(dummies)
library(caret)
library(pROC)
library(Rcpp)

#Data exploration and combination
test.import <- read.csv(file.choose())
summary(test.import )
train.import <- read.csv(file.choose())
summary(train.import)
full.import <- bind_rows(train.import,test.import)
summary(full.import)
str(full.import)
colSums(is.na(full.import))

## Remarks:
### Test PassengerId=892to1309 Train PassengerId=1to891
### Count of NAs - Survived=418 (in test data only), Age=263, Fare=1
### Not sure how to engineer Cabin

# Data imputation
full.import.exc.survival = full.import[,!names(full.import) %in% "Survived"]
md.pattern(full.import.exc.survival)

full.imputed <- complete(mice(full.import.exc.survival,m=5,meth='pmm',seed=1),1)
md.pattern(full.imputed)
colSums(is.na(full.imputed))
full.imputed$Survived <- full.import$Survived
str(full.imputed)
summary(full.imputed)


# Feature Engineering
##1 SibSP and Parch can be consolidate into a new FamSize column AND Separate Lone Travelers from Family travelers
full.imputed$FamSize <- full.imputed$SibSp+full.imputed$Parch+1
full.imputed$FamAbroad <- ifelse(full.imputed$SibSp+full.imputed$Parch > 0 ,"1","0")


##2 Ticket Frequency
tableTickets <- table(c(as.character(full.imputed$Ticket)))
full.imputed$TicketFreq <- tableTickets[as.character(full.imputed$Ticket)]

##3 Divide into Age groups
full.imputed$Age_0to20 <- ifelse(full.imputed$Age < 21 ,"1","0")
full.imputed$Age_21to40 <- ifelse(full.imputed$Age > 20 & full.imputed$Age < 41  ,"1","0")
full.imputed$Age_41to60 <- ifelse(full.imputed$Age > 40 & full.imputed$Age < 61  ,"1","0")
full.imputed$Age_61to80 <- ifelse(full.imputed$Age > 60 & full.imputed$Age < 81  ,"1","0")

##4 Fare classes
full.imputed$FareGroup <- with(full.imputed, cut(Fare, breaks=c(0,7.9,32,600),right=FALSE,labels=c('low','medium','high')))

##5 Cabin passenger or not
full.imputed$CabinNumExist <- ifelse(full.imputed$Cabin > 0,"1","0")

##6 Converting "Survived", "Pclass","Sex","FamSize","FamAbroad", "Embarked","TicketFreq" into factors
for (i in c("Survived", "Pclass","Sex","FamSize","FamAbroad", "Embarked","TicketFreq","Age_0to20","Age_21to40","Age_41to60","Age_61to80","CabinNumExist")){
  full.imputed[,i]=as.factor(full.imputed[,i])
}
str(full.imputed)

##7 Drop "Name", "Ticket" and "Cabin"
drop <- c("Name", "Ticket" , "Cabin") ## Note*. Cabin 0/1 + ## and Age range 
full.imputed = full.imputed[,!names(full.imputed) %in% drop]

##8 Create dummy variables for categorical variables
full.imputed <- dummy.data.frame(full.imputed, names=c("Pclass","Sex","FamSize","Embarked","TicketFreq","FareGroup"), sep="_")

#data checking
summary(full.imputed)
str(full.imputed)
head(full.imputed)

#Run GLM for prelim preview
glm.full <- glm(Survived ~ ., family = binomial, full.imputed)
summary(glm.full)

#Split back to test and train data sets
test.data <- full.imputed[which(full.imputed$PassengerId>891),]
train.data <- full.imputed[which(full.imputed$PassengerId<892),]
#write.csv(test.data,file="test_data.csv",row.names = FALSE)
#write.csv(train.data,file="train_data.csv",row.names = FALSE)
str(test.data)
str(train.data)

# Model creation
logr.model <- glm(Survived ~ ., family = binomial(link = "logit"), train.data)
summary(logr.model)

# Confusion Matrix
actualsurvival <- train.data$Survived
train.result<- predict(logr.model,newdata=train.data,type='response')
train.result<- as.factor(ifelse(train.result > 0.5 ,1,0))
confusionMatrix(data=train.result, reference=actualsurvival)

# ROC Curve and AUC
test_prob = predict(logr.model,newdata=train.data,type='response')
test_roc = roc(actualsurvival ~ test_prob, plot = TRUE, print.auc = TRUE)

# Test data prediction and result population
result <- data.frame(predict(logr.model,newdata=test.data,type='response'))
submission <- data.frame(test.data$PassengerId,result)
colnames(submission) <- c('PassengerId','Survived')
submission$Survived <- as.integer(ifelse(submission$Survived > 0.5 ,1,0)) # use 0.5 as threshold
write.csv(submission,file="submission_v3.csv", row.names = FALSE)


#Create model matrix
full.imputed.exl.survived<-subset(full.imputed,select=-Survived)
X <- model.matrix(~., full.imputed.exl.survived)
Y <- as.numeric(as.character(train.data$Survived))

#Split training dataset to generate a testing set with Survived
X.training<-X[1:891,]
X.training<-X[1:668,]
X.testing<-X[669:891,]
X.prediction<-X[892:1309,]
Y.training<-Y[1:891]
Y.training<-Y[1:668]
Y.testing<-Y[669:891]

# Run Lasso
lasso.fit<-glmnet(x = X.training, y = Y.training, alpha = 1)
plot(lasso.fit, xvar = "lambda")
crossval <-  cv.glmnet(x = X.training, y = Y.training, alpha = 1,family="binomial",type.measure = "mse") 
plot(crossval)
penalty.lasso <- crossval$lambda.min


#best value of lambda
lambda_1se <- crossval$lambda.1se

#regression coefficients
coef(crossval,s=lambda_1se)


lasso_prob <- predict(crossval,newx = X.training ,s=lambda_1se,type="response")

#translate probabilities to predictions
lasso_predict <- rep(0, length(Y.training))
lasso_predict[lasso_prob > 0.5] <- 1 
confusion.matrix<- table(pred=lasso_predict,true=Y.training)

#accuracy
mean(lasso_predict==Y.training)

Count.correct<-confusion.matrix[1,1]+confusion.matrix[2,2]
Count.wrong<-confusion.matrix[1,2]+confusion.matrix[2,1]

Accuracy.rate<-Count.correct/(Count.correct+Count.wrong)

test_roc = roc(Y.training ~ lasso_prob, plot = TRUE, print.auc = TRUE)

#lasso testing 
lasso_prob <- predict(crossval,newx = X.testing,s=lambda_1se,type="response")
lasso_predict <- rep(0, length(Y.testing))
lasso_predict[lasso_prob > 0.5] <- 1 

confusion.matrix<- table(pred=lasso_predict,true=Y.testing)

#accuracy
mean(lasso_predict==Y.testing)
 
Count.correct<-confusion.matrix[1,1]+confusion.matrix[2,2]
Count.wrong<-confusion.matrix[1,2]+confusion.matrix[2,1]

Accuracy.rate<-Count.correct/(Count.correct+Count.wrong)

test_roc = roc(Y.testing ~ lasso_prob, plot = TRUE, print.auc = TRUE)

#Prediction
lasso_predict<-predict(crossval, s = penalty.lasso, newx =X.prediction)
lasso_predict[lasso_predict > 0.5] <- 1
lasso_predict[lasso_predict <= 0.5] <- 0
colnames(lasso_predict) <- "Survived"
lasso_predict<-as.data.frame(lasso_predict)

write.csv(lasso_predict, file = "Lasso Titanic Survival.csv") 

#Using Ridge
ridge.fit<-glmnet(x = X.training, y = Y.training, alpha = 0)
crossval.ridge <-  cv.glmnet(x = X.training, y = Y.training, alpha = 0, family="binomial",type.measure = "mse")
penalty.ridge <- crossval.ridge$lambda.min 

lambda_1se <- crossval.ridge$lambda.1se

ridge_prob <- predict(crossval.ridge,newx = X.training,s=lambda_1se,type="response")
ridge_predict <- rep(0, length(Y.training))
ridge_predict[ridge_prob > 0.5] <- 1 # threshold needs to be determined

#confusion matrix
confusion.matrix<- table(pred=ridge_predict,true=Y.training)

#accuracy
mean(ridge_predict==Y.training)

Count.correct<-confusion.matrix[1,1]+confusion.matrix[2,2]
Count.wrong<-confusion.matrix[1,2]+confusion.matrix[2,1]

Accuracy.rate<-Count.correct/(Count.correct+Count.wrong)

test_roc = roc(Y.training ~ ridge_prob, plot = TRUE, print.auc = TRUE)

#ridge testing 
ridge_prob <- predict(crossval,newx = X.testing,s=lambda_1se,type="response")
ridge_predict <- rep(0, length(Y.testing))
ridge_predict[ridge_prob > 0.5] <- 1 

confusion.matrix<- table(pred=ridge_predict,true=Y.testing)

#accuracy
mean(ridge_predict==Y.testing)

Count.correct<-confusion.matrix[1,1]+confusion.matrix[2,2]
Count.wrong<-confusion.matrix[1,2]+confusion.matrix[2,1]

Accuracy.rate<-Count.correct/(Count.correct+Count.wrong)

test_roc = roc(Y.testing ~ ridge_prob, plot = TRUE, print.auc = TRUE)

#Prediction
ridge_predict<-predict(crossval, s = penalty.lasso, newx =X.prediction)
ridge_predict[ridge_predict > 0.5] <- 1
ridge_predict[ridge_predict <= 0.5] <- 0
colnames(ridge_predict) <- "Survived"
ridge_predict<-as.data.frame(ridge_predict)
  
write.csv(ridge_predict, file = "Lasso Titanic Survival.csv")

#Remove Testing Data

#Create model matrix
full.imputed.exl.survived<-subset(full.imputed,select=-Survived)
X <- model.matrix(~., full.imputed.exl.survived)
Y <- as.numeric(as.character(train.data$Survived))

#Split training dataset to generate a testing set with Survived
X.training<-X[1:891,]
#X.training<-X[1:668,]
#X.testing<-X[669:891,]
X.prediction<-X[892:1309,]
Y.training<-Y[1:891]
#Y.training<-Y[1:668]
#Y.testing<-Y[669:891]

# Run Lasso
lasso.fit<-glmnet(x = X.training, y = Y.training, alpha = 1)
plot(lasso.fit, xvar = "lambda")
crossval <-  cv.glmnet(x = X.training, y = Y.training, alpha = 1,family="binomial",type.measure = "mse") 
plot(crossval)
penalty.lasso <- crossval$lambda.min


#best value of lambda
lambda_1se <- crossval$lambda.1se

#regression coefficients
coef(crossval,s=lambda_1se)


lasso_prob <- predict(crossval,newx = X.training ,s=lambda_1se,type="response")

#translate probabilities to predictions
lasso_predict <- rep(0, length(Y.training))
lasso_predict[lasso_prob > 0.5] <- 1 
confusion.matrix<- table(pred=lasso_predict,true=Y.training)

#accuracy
mean(lasso_predict==Y.training)

Count.correct<-confusion.matrix[1,1]+confusion.matrix[2,2]
Count.wrong<-confusion.matrix[1,2]+confusion.matrix[2,1]

Accuracy.rate<-Count.correct/(Count.correct+Count.wrong)

test_roc = roc(Y.training ~ lasso_prob, plot = TRUE, print.auc = TRUE)

#Prediction
lasso_predict<-predict(crossval, s = penalty.lasso, newx =X.prediction)
lasso_predict[lasso_predict > 0.5] <- 1
lasso_predict[lasso_predict <= 0.5] <- 0
colnames(lasso_predict) <- "Survived"
lasso_predict<-as.data.frame(lasso_predict)

write.csv(lasso_predict, file = "Lasso Titanic Survival.csv") 

#Using Ridge
ridge.fit<-glmnet(x = X.training, y = Y.training, alpha = 0)
crossval.ridge <-  cv.glmnet(x = X.training, y = Y.training, alpha = 0, family="binomial",type.measure = "mse")
penalty.ridge <- crossval.ridge$lambda.min 

#best value of lambda
lambda_1se_ridge <- crossval.ridge$lambda.1se

#regression coefficients
coef(crossval.ridge,s=lambda_1se)

ridge_prob <- predict(crossval.ridge,newx = X.training,s=lambda_1se,type="response")
ridge_predict <- rep(0, length(Y.training))
ridge_predict[ridge_prob > 0.5] <- 1 # threshold needs to be determined

#confusion matrix
confusion.matrix<- table(pred=ridge_predict,true=Y.training)

#accuracy
mean(ridge_predict==Y.training)

Count.correct<-confusion.matrix[1,1]+confusion.matrix[2,2]
Count.wrong<-confusion.matrix[1,2]+confusion.matrix[2,1]

Accuracy.rate<-Count.correct/(Count.correct+Count.wrong)

test_roc = roc(Y.training ~ ridge_prob, plot = TRUE, print.auc = TRUE)

#Prediction
ridge_predict<-predict(crossval, s = penalty.ridge, newx =X.prediction)
ridge_predict[ridge_predict > 0.5] <- 1
ridge_predict[ridge_predict <= 0.5] <- 0
colnames(ridge_predict) <- "Survived"
ridge_predict<-as.data.frame(ridge_predict)

write.csv(ridge_predict, file = "Ridge Titanic Survival.csv")

