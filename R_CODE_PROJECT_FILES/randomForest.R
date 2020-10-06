#loading packages
library(ranger)
library(caret)
library(ROCR)
library(e1071)
library(doParallel)
library(randomForest)
#using all cores
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

##########################################################
#splitting data set to train and test set
split_index <- createDataPartition(titan_clean$Survived, p=0.8, list = FALSE)
testing_set <- titan_clean[-split_index,]
training_set <- titan_clean[split_index,]

#tunning random forest
nodesize.vals <- c(2, 3, 4, 5)
ntree.vals <- c(200, 500, 1000, 2000)
tuning.results <- tune.randomForest(Survived~., 
                                    data = training_set,
                                    mtry=3, 
                                    nodesize=nodesize.vals,
                                    ntree=ntree.vals)
#saving tuned result as rds
saveRDS(tuning.results, "./tuning.results.rds")
print(tuning.results)

#training random forest using R Recipes
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
randfor_tuned <- train(Survived~., data = training_set, method = "rf", 
                       metric = "Accuracy", ntree = 2000)
#print trained random forest
print(randfor_tuned)
#gettting details of final model
print(randfor_tuned$finalModel)
randfor_tuned$results
#saving trained model
saveRDS(randfor_tuned, "./randfor_tuned.rds")
#using trained model recipe result
#to run randomforest model
set.seed(127)
final_modelN <- randomForest(Survived~.,training_set, mtry = 2, ntree = 2000)
#confusion matrix 
predictions <- predict(final_modelN, newdata = testing_set)
confusionMatrix(predictions, testing_set$Survived)
########################################################################
#using different mtry and ntree to run random forest
set.seed(127)
final_model3 <- randomForest(Survived~.,training_set, mtry = 1, ntree = 1800)
saveRDS(final_model3, "./inal_model_rf.rds")
rf.model <- readRDS("./inal_model_rf.rds")
print(rf.model)
predictions <- predict(rf.model, newdata = testing_set)
confusionMatrix(predictions, testing_set$Survived)

#plotting ROC Curve.
rf.preds.values <- predict(rf.model, testing_set[,-1],
                             type = "prob")
rf.predictions.values <- rf.preds.values[,2]
predictions <- prediction(rf.predictions.values,
                          testing_set$Survived)
par(mfrow=c(1,2))
plot.roc.curve(predictions, title.text="Random Forest ROC Curve")
plot.pr.curve(predictions, title.text="Random Forest Precision/Recall Curve")

