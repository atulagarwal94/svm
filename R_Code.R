############################ SVM Digit Recogniser #################################
# 1. Business Understanding
# 2. Data Understanding
# 3. Data Preparation
# 4. Model Building 
#  4.1 Linear kernel
#  4.2 RBF Kernel
# 5 Hyperparameter tuning and cross validation

#####################################################################################

# 1. Business Understanding: 

#The objective is to identify each of a large number of
# rectangular pixel displays as one of the 10 digits

#####################################################################################

# 2. Data Understanding:
# Number of Instances: 60,000
# Number of Attributes: 785 

#3. Data Preparation: 


#Loading Neccessary libraries
library('e1071')
library(kernlab)
library(readr)
library(e1071)


#Loading Data

Data <- read.csv("mnist_train.csv" , header = FALSE)
colnames(Data)[1] <- "letter"

testData <- read.csv("mnist_test.csv" , header = FALSE)
colnames(testData)[1] <- "letter"

#Understanding Dimensions

dim(Data)

#Structure of the dataset

str(Data)

#printing first few rows

head(Data)

#Exploring the data

summary(Data)

#checking missing value

sapply(Data, function(x) sum(is.na(x)))


#Making our target class to factor

Data$letter<-factor(Data$letter)


# Split the data into train and test set

set.seed(1)
train.indices = sample(1:nrow(Data), 1500)
train = Data[train.indices, ]
test.indices = sample(1:nrow(Data), 1500)
test = Data[test.indices, ]



#Constructing Model

#Using Linear Kernel
Model_linear <- ksvm(letter~ ., data = train, scale = FALSE, kernel = "vanilladot")
Eval_linear<- predict(Model_linear, test)

#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear,test$letter)

# Accuracy 0.8860 for 1500 rows
# Accuracy 0.9077 for 3000 rows
# Accuracy 0.9128 for 6000 rows
# Accuracy 0.9282 for 10000 rows

#Using Polynomial Kernel
Model_polynomial <- ksvm(letter~ ., data = train, scale = FALSE, kernel = "polydot", C=1)
Eval_polynomial<- predict(Model_polynomial, test)

#confusion matrix - Polynomial Kernel
confusionMatrix(Eval_polynomial,test$letter)

# Accuracy 0.8860 for 1500 rows
# Accuracy 0.9077 for 3000 rows
# Accuracy 0.9128 for 6000 rows
# Accuracy 0.9282 for 10000 rows

#Using RBF Kernel
Model_RBF <- ksvm(letter~ ., data = train, scale = FALSE, kernel = "rbfdot")
Eval_RBF<- predict(Model_RBF, test)

#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF,test$letter)

# Accuracy 0.9093 for 1500 rows
# Accuracy 0.9077 for 3000 rows
# Accuracy 0.9485 for 6000 rows
# Accuracy 0.9615 for 6000 rows

############   Hyperparameter tuning and Cross Validation #####################

# We will use the train function from caret package to perform Cross Validation. 

#traincontrol function Controls the computational nuances of the train function.
# i.e. method =  CV means  Cross Validation.
#      Number = 2 implies Number of folds in CV.

trainControl <- trainControl(method="cv", number=5)


# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.

set.seed(7)
grid <- expand.grid(.sigma=c(0.01, 0.05), .C=c(0.1, 0.5 ,1,2) )


#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svm <- train(letter~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

print(fit.svm)

plot(fit.svm)

# With 2000 rows of training data
# Resampling results across tuning parameters:

#sigma  C    Accuracy   Kappa
#0.01   0.1  0.1179992  0    
#0.01   0.5  0.1179992  0    
#0.01   1.0  0.1179992  0    
#0.01   2.0  0.1179992  0    
#0.05   0.1  0.1179992  0    
#0.05   0.5  0.1179992  0    
#0.05   1.0  0.1179992  0    
#0.05   2.0  0.1179992  0    

#The final values used for the model were sigma = 0.05 and C = 0.1.

######################## IMPORTANT #############################

# SINCE tuning parameters gave very low accuracy even with different amount of training data i.e 2k , 4k , 6k
# and the result was similar to one the printed above I manually calculated with varying C and sigma values
# which also didnt helped much
# So I went by normal RBF Kernel with default parameters giving an accuracy of 96%

#######################
##### FINAL MODEL #####
#######################

# RBF Kernel With Accuracy of 96%
Model_RBF