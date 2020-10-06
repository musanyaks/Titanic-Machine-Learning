library(dplyr)
library(gbm)
library(xgboost)
library(doParallel)
#making use of all coresd
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

#######################################################################
#splitting data into test and training set
split_index <- createDataPartition(titan_clean$Survived, p=0.8, list = FALSE)
testing_set <- titan_clean[-split_index,]
training_set <- titan_clean[split_index,]

#tunung gbm model
grid = expand.grid(.n.trees=seq(100,500, by=200), 
                  .interaction.depth=seq(1,4, by=1), .shrinkage=c(.001,.01,.1),
                  .n.minobsinnode=10)
control = trainControl(method="CV", number=10)
#traing gbm model
set.seed(123)
titan_gbm_train <- train(Survived~., data=training_set, method="gbm",  
                         trControl=control, tuneGrid=grid, metric = "Accuracy")
#printing gbm results
titan_gbm_train
#saving the gbm model results
saveRDS(titan_gbm_train,"./titan_gbm_train.rds")
#loading saved gbm model
set.seed(127)
gbm.model <- readRDS("./titan_gbm_train.rds")
print(gbm.model)
#confusion matrix
predictions <- predict(gbm.model, newdata = testing_set)
confusionMatrix(predictions, testing_set$Survived)

#plotting ROC Curve
gbm.preds.values <- predict(gbm.model, testing_set[,-1],
                                 type = "prob")
gbm.predictions.values <- gbm.preds.values[,2]
predictions <- prediction(gbm.predictions.values,
                          testing_set$Survived)
par(mfrow=c(1,2))
plot.roc.curve(predictions, title.text="GBBOOST  ROC Curve")
plot.pr.curve(predictions, title.text="GBBOOST  Precision/Recall Curve")

nnet.model
