# install.packages(c("dplyr", "ggplot2", "ISLR", "MASS", "glmnet", "randomForest",
# "gbm", "rpart", "boot"))

library(dplyr)
library(ggplot2)
library(ISLR)
library(MASS)
library(glmnet)
library(randomForest)
library(gbm)
library(rpart)
library(boot)

# Import Data Set
# Bank Marketing Data Set

df <- read.csv('D:/EUN/bank-additional-full.csv', header=TRUE,sep=";")

head(df)

glimpse(df)

# Encoding

df$y <- ifelse(df$y == "yes", 1, 0)

library(caret)

onehot <- dummyVars(~., data = df)
df_new <- data.frame(predict(onehot, newdata = df))

# Check missing values

sum(is.na(df_new))

# Visualization(age)

training %>%
  ggplot(aes(age, group = y, colour=factor(y))) +
  geom_density()

# Visualization(duration)

training %>%
  ggplot(aes(duration,  group = y, colour=factor(y))) +
  geom_density()

# Divide the training group and the test group

set.seed(1000)

n <- nrow(df_new)

idx <- 1:n
training_idx <- sample(idx, n * .60, replace = TRUE)
idx <- setdiff(idx, training_idx)
test_idx <- sample(idx, n * .40)



length(training_idx)
length(test_idx)

training <- df_new[training_idx,]
test <- df_new[test_idx,]

#Logistic Regression
# Fitting Logistic Regression Model
glm_full <- glm(y ~., data = training, family = binomial("logit"))

summary(glm_full)

# Select significant variables

glm_selected <- glm(y~ age + contact.cellular + month.aug + month.jun + month.mar + 
                      month.may + month.nov + day_of_week.fri + day_of_week.mon + 
                      month.nov + day_of_week.fri +day_of_week.mon + day_of_week.thu + 
                      duration + campaign + poutcome.failure + poutcome.nonexistent +
                      emp.var.rate + cons.price.idx + euribor3m , data = training, 
                    family = binomial("logit"))

library(car)

vif(glm_selected)

# Remove Multicollinearity

glm_model <- glm(y~ age + contact.cellular + month.aug + month.jun + month.mar +
                   month.may + month.nov + day_of_week.fri + day_of_week.mon + 
                   month.nov + day_of_week.fri +day_of_week.mon + day_of_week.thu + 
                   duration + campaign + poutcome.failure + poutcome.nonexistent,
                 data = training, family = binomial("logit"))

vif(glm_model)

# Bonferonni's significance probability is greater than the significance level of 0.05.
# So there is no outlier.

outlierTest(glm_model)

# Predict GLM

yhat_lm <- predict(glm_model, newdata = test, type = 'response')

library(gridExtra)

p1 <- ggplot(data.frame(test$y, yhat_lm), aes(test$y, yhat_lm, group=test$y,
                                              fill=factor(test$y))) + 
  geom_boxplot()

p2 <-ggplot(data.frame(test$y, yhat_lm), aes(yhat_lm, fill=factor(test$y))) + 
  geom_density(alpha = .5)

grid.arrange(p1, p2, ncol =2)

# install.packages("ROCR")

library(ROCR)

# GLM Model Evaluation

pred_lm <-prediction(yhat_lm, test$y)
perf_lm <- performance(pred_lm, measure = 'tpr', x.measure = 'fpr')
performance(pred_lm, 'auc')@y.values[[1]]

rmse_lm <- RMSE(test$y, yhat_lm)
rmse_lm


# Glmnet
# Fitting Lasso Model

xx <- model.matrix(y~ ., df_new)
x <- xx[training_idx, ]
dim(x)

cv_fit <- cv.glmnet(x, training$y, family = 'binomial')
plot(cv_fit)

length(which(coef(cv_fit, s = "lambda.min")>0))
length(which(coef(cv_fit, s = "lambda.1se")>0))

# Select alpha

set.seed(1000)

foldid <- sample(1:10, size=length(training$y), replace = TRUE)
cv1 <- cv.glmnet(x, training$y, foldid = foldid, alpha = 1, family = 'binomial')
cv.5 <- cv.glmnet(x, training$y, foldid = foldid, alpha = .5, family = 'binomial')
cv0 <- cv.glmnet(x, training$y, foldid = foldid, alpha = 0, family = 'binomial')

par(mfrow=c(2,2))
plot(cv1, main = "Alpha = 1")
plot(cv.5, main = "Alpha = 0.5")
plot(cv0, main = "Alpha = 0")

# Predict Glmnet

predict(cv_fit, s="lambda.1se", newx = x[1:5,], type = 'response')

lasso_coef <-predict(cv_fit, s="lambda.1se", newx = x[1:5,], type = 'coefficients')

lasso_coef[lasso_coef!=0]

yhat_glmnet <- predict(cv_fit, s="lambda.1se", newx = xx[test_idx,], type= 'response')
yhat_glmnet <- yhat_glmnet[,1]

# Glmnet Model Evlauation
pred_glmnet <- prediction(yhat_glmnet, test$y)
perf_glmnet <- performance(pred_glmnet, measure = 'tpr', x.measure = 'fpr')

performance(pred_glmnet, "auc")@y.values[[1]]

rmse_glmnet <- RMSE(test$y, yhat_glmnet)
rmse_glmnet

# Fitting Decision Tree Model

df_tree <- rpart(y ~ ., data=training)
df_tree

operation <- par(mfrow = c(1,1), xpd = NA)
plot(df_tree)
text(df_tree, use.n = TRUE)
par(operation)

# Pruning Decision Tree

# install.packages("tree")
library(tree)

plotcp(df_tree)

prune_df_tree<-prune(df_tree, cp= df_tree$cptable[which.min(df_tree$cptable[,"xerror"]),"CP"])
plot(prune_df_tree)
text(prune_df_tree)

# Predict Decision Tree
yhat_tree <- predict(prune_df_tree, test, type='vector')

# Decision Tree Model Evlauation
pred_tree <- prediction(as.numeric(yhat_tree), as.numeric(test$y))
perf_tree <- performance(pred_tree, measure = 'tpr', x.measure = 'fpr')

performance(pred_tree, "auc")@y.values[[1]]

rmse_tree <- RMSE(as.numeric(test$y),as.numeric(yhat_tree))
rmse_tree

# Random Forest
# Fitting Randomforest Model

training$y <- as.factor(training$y)

set.seed(1000)

model_rf_full <- randomForest(y ~., training)
model_rf_full

plot(model_rf_full)

# Variable Importance

varImpPlot(model_rf_full)

important_values <- names(model_rf_full$importance[model_rf_full$importance > 100,])
important_values

# Re-fit to Variable Importance

model_rf <- randomForest(y ~ age + duration + campaign + pdays + poutcome.success + emp.var.rate
                         + cons.price.idx + cons.conf.idx + euribor3m + nr.employed, training)

training<-training[,c(important_values, 'y')]
test<-test[,c(important_values, 'y')]

# Predict Randomforest
yhat_rf <-predict(model_rf, newdata = test, type = 'class')

# Randomforest Model Evluation

pred_rf <- prediction(as.numeric(yhat_rf), as.numeric(test$y))
perf_rf <- performance(pred_rf, measure = 'tpr', x.measure = 'fpr')

performance(pred_rf, "auc")@y.values[[1]]

rmse_rf <- RMSE(as.numeric(test$y), as.numeric(yhat_rf))
rmse_rf

# Compare the Models
# RMSE

data.frame(lm = rmse_lm,
           glmnet = rmse_glmnet,
           tree = rmse_tree,
           rf = rmse_rf) %>%
  reshape2::melt(value.name =  'RMSE', variable.name = 'METHOD')

# ROC curve

plot(perf_lm, col = 'blue', main = 'ROC curve')
plot(perf_glmnet, add=TRUE, col = 'yellow')
plot(perf_tree, add=TRUE, col='red')
plot(perf_rf, add=TRUE, col='black')
abline(0,1)
legend('bottomright', inset = .1,
       legend = c('LM', 'glmnet', 'tree', 'RF'),
       col=c('blue', 'yellow', 'red', 'black'), lty=1, lwd=2)