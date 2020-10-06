#loading R packages
library(caret)
library(doParallel)
#loading dataset
df <- read.csv("titan.csv")
df <- df[,-c(1)]
#View(head(df))
df_clean <- df[,-c(1,4,9)]
#View(head(df_clean))
str(df_clean)

# data transformation
to.factor <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.factor(df[[variable]])
  }
  return(df)
}

cate_variable <- c("Pclass","Sex",  "Embarked", "Title", "Survived")
df_clean <- to.factor(df = df_clean, variables = cate_variable)
str(df_clean)

#View(head(df_clean))

#normalizing - scaling
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
df_clean2 <- as.data.frame(lapply(df_clean[,c(4,7)], normalize))
titan_clean <- as.data.frame(cbind(df_clean[-c(4,7)], df_clean2))


#splitting data into train and test set
split_index <- createDataPartition(titan_clean$Survived, p=0.8, list = FALSE)
testing_set <- titan_clean[-split_index,]
training_set <- titan_clean[split_index,]
#View(head(titan_clean))
write.csv(titan_clean, file = "titan_normalized.csv")

#training the model
set.seed(123)
control <- trainControl(method = "cv", number = 10)
training_set$FamilySize <- as.numeric(training_set$FamilySize)
fit_logist <- train(Survived~., data = training_set, method = "glm", 
                    metric = "Accuracy", trControl = control)
#printing model results
print(fit_logist)
print(fit_logist$finalModel)

#saving model result
saveRDS(fit_logist, "./logistic.model.rds")
#loading saved model
trained.log.model <- readRDS("./logistic.model.rds")
print(trained.log.model)
print(trained.log.model$finalModel)
#confusion matrix
set.seed(127)
predictions <- predict(trained.log.model, newdata = testing_set)
confusionMatrix(predictions, testing_set$Survived)
summary(fit_logist)


#plotting ROC Curve.

logistic.preds.values <- predict(trained.log.model, testing_set[,-1],
                                 type = "prob")
logistic.predictions.values <- logistic.preds.values[,2]
predictions <- prediction(logistic.predictions.values,
                          testing_set$Survived)
par(mfrow=c(1,2))
plot.roc.curve(predictions, title.text="Logistic Regression ROC Curve")
plot.pr.curve(predictions, title.text="Logistic Regression Precision/Recall Curve")
