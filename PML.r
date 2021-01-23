#https://rpubs.com/jordanlsmith/pmlproject

library(caret)
#for Random Forest model
library(randomForest)
library(e1071)
#for GBM model
library(gbm)
library(survival)
library(splines)
library(parallel)
library(plyr)
#for LDA model
library(MASS)
#import data, removing any 'DIV/0', NA, and empty values
train <- read.csv(file = "pml-training.csv", 
                  na.strings=c('#DIV/0', '', 'NA', '#DIV/0!'))
validation <- read.csv(file = "pml-testing.csv",
                 na.strings=c('#DIV/0', '', 'NA','#DIV/0!')) 
#here, I will split the training data into training and test sets and use the
#given test set as validation
inTrain = createDataPartition(train$classe, p = .7, list = FALSE)
training = train[ inTrain,]
testing = train[-inTrain,]

#remove columns 1:7 (which include non-movement information)
train_filter <- training[, -c(1:7)]
testing_filter <-testing[, -c(1:7)]
validation_filter <- validation[, -c(1:7)]
#remove columns with >80% NA
train_na_filter <- train_filter[, colSums(is.na(train_filter)) < nrow(train_filter) * 0.8]
testing_na_filter <-testing_filter[, colSums(is.na(train_filter)) < nrow(train_filter) * 0.8]
validation_na_filter <- validation_filter[, colSums(is.na(train_filter)) < nrow(train_filter) * 0.8]
#preProcess data to remove near-zero variance classifiers, then center and scale
#remaining classifiers
train_center_scale_nzv <- preProcess(train_na_filter, 
                                     method = c("center", "scale", "nzv"))
#Pre-processing:
#  - centered (52)
#  - ignored (1)
#  - scaled (52)

#apply preProcess to produce training data for modeling
modelTraining <- predict(train_center_scale_nzv, newdata = train_na_filter)
testingFinal <- predict(train_center_scale_nzv, newdata = testing_na_filter)
validationFinal <- predict(train_center_scale_nzv, newdata = validation_na_filter)

set.seed(100000)
#prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)

#train the random forest model
rfModel <- train(classe ~ ., 
                 method = "rf", 
                 data = modelTraining, trControl=control)
#train the GBM model
gbmModel <- train(classe ~ ., 
                  method = "gbm", 
                  data = modelTraining, trControl=control, verbose = FALSE)
#train the LDA model
ldaModel <- train(classe ~ ., 
                  method = "lda", 
                  data = modelTraining, trControl=control)

#predict, using the three models
rfPred <- predict(rfModel, testingFinal)
gbmPred <- predict(gbmModel, testingFinal)
ldaPred <- predict(ldaModel, testingFinal)

#combine three prediction sets into one df
ComboDF <- data.frame(rfPred, gbmPred, ldaPred, classe=testingFinal$classe)

#train stacked Random Forest model
comboModel <-train(classe ~ ., 
                   method = "rf", 
                   data = ComboDF, trControl=control)

#combo model predict
comboPred <- predict(comboModel, testingFinal)

#finding accuracies for each model
rfCM <- confusionMatrix(testingFinal$classe, rfPred)$overall['Accuracy']
gbmCM <- confusionMatrix(testingFinal$classe, gbmPred)$overall['Accuracy']
ldaCM <- confusionMatrix(testingFinal$classe, ldaPred)$overall['Accuracy']
comboCM <- confusionMatrix(testingFinal$classe, comboPred)$overall['Accuracy']

#make an accuracy summary table
AccuracyResults <- data.frame(
  Model = c('RF', 'GBM', 'LDA', 'Combo'),
  Accuracy = rbind(rfCM, gbmCM, ldaCM, comboCM)
)
#Print Accuracy table
AccuracyResults
