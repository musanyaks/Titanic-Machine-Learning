library(rpart.plot)
library(rpart)
library(caret)
library(ROCR)
library(e1071)

 
source("plot_utils.R")
#splitting data into test and training set
split_index <- createDataPartition(titan_clean$Survived, p=0.8, list = FALSE)
testing_set <- titan_clean[-split_index,]
training_set <- titan_clean[split_index,]

#tuning cart model

trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "Accuracy"
fit.cart <- train(Survived~., data=training_set, 
                  method="rpart", metric=metric, trControl=trainControl)
#saving model
saveRDS(fit.cart, "./fit_cart.rds")
print(fit.cart)#print result
cart.model <- readRDS("./fit_cart.rds")#loading model
print(cart.model)
#confusion matrix
set.seed(127)
predictions <- predict(cart.model, newdata = testing_set)
confusionMatrix(predictions, testing_set$Survived)


#plotting ROC Curve
cart.preds.values <- predict(cart.model, testing_feature_variables,
                                 type = "prob")
cart.predictions.values <- cart.preds.values[,2]
predictions <- prediction(cart.predictions.values,
                          testing_set$Survived)
par(mfrow=c(1,2))
plot.roc.curve(predictions, title.text="CART ROC Curve")
plot.pr.curve(predictions, title.text="CART Precision/Recall Curve")

