#loading package for modelling
library(caret)

  #############################################################################
#splitting the data set to test and training set.  
split_index <- createDataPartition(titan_clean$Survived, p=0.8, list = FALSE)
testing_set <- titan_clean[-split_index,]
training_set <- titan_clean[split_index,]
#training nnet model using R Recipe.
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "Accuracy"
fit.nnet <- train(Survived~., data=training_set, 
                   method="nnet", metric=metric, trControl=trainControl)  

#saving trained model
saveRDS(fit.nnet, "./fit_nnet.rds")
#loading saved model
nnet.model <- readRDS("./fit_nnet.rds")
print(nnet.model)
#getting confusion matrix results
set.seed(127)
predictions <- predict(nnet.model, newdata = testing_set)
confusionMatrix(predictions, testing_set$Survived)

#plotting ROC Curve
nnet.preds.values <- predict(nnet.model, testing_set[,-1],
                            type = "prob")
nnet.predictions.values <- nnet.preds.values[,2]
predictions <- prediction(nnet.predictions.values,
                          testing_set$Survived)
par(mfrow=c(1,2))
plot.roc.curve(predictions, title.text="Neural Network ROC Curve")
plot.pr.curve(predictions, title.text="Neural Network Precision/Recall Curve")



