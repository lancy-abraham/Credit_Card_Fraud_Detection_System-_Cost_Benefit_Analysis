library(dplyr)
library(ranger)     
library(caret) 
library(caTools)
library(data.table)
library(ggplot2)
library(corrplot)
library(Rtsne)
library(ROSE) 
library(pROC)
library(rpart)
library(rpart.plot)
library(Rborist)
library(xgboost)

# importing the dataset
dataset <- setDT(read.csv('C:/Users/Lancy/Desktop/IIMK/4. DS Strategy and Capstone/4. Module 1 - Capstone Project - Credit Card Fraud Detection/Project Details/Credit Card Fraud Analysis & Cost Benefit Analysis- Lancy/creditcard.csv'))

# exploring the credit card data
head(dataset)
tail(dataset)

# view the table from class column (0 for legit transactions and 1 for fraud)
table(dataset$Class)

# view names of colums  of dataset
names(dataset)

# view summary of amount and histogram
summary(dataset$Amount)
hist(dataset$Amount)
hist(dataset$Amount[dataset$Amount < 100])
hist(dataset$Time)
cor(dataset$Time, dataset$Amount)

# view variance and standard deviation of amount column
var(dataset$Amount)
sd(dataset$Amount)

# check whether there are any missing values in colums
colSums(is.na(dataset))

# visualizing the distribution of transcations across time
dataset %>%
  ggplot(aes(x = Time, fill = factor(Class))) + 
  geom_histogram(bins = 100) + 
  labs(x = "Time elapsed since first transcation (seconds)", y = "no. of transactions", title = "Distribution of transactions across time") +
  facet_grid(Class ~ ., scales = 'free_y') + theme()

# visualizing the distribution of transcations amount by class
p <- ggplot(dataset, aes(x = Class, y = Amount)) + geom_boxplot() + ggtitle("Distribution of transaction amount by class")
print(p)

# correlation of anonymous variables with amount and class
correlation <- cor(dataset[, -1], method = "pearson")
corrplot(correlation, number.cex = 1, method = "color", type = "full", tl.cex=0.7, tl.col="black")

# only use 10% of data to compute SNE and perplexity to 20
tsne_data <- 1:as.integer(0.1*nrow(dataset))
tsne <- Rtsne(dataset[tsne_data,-c(1, 31)], perplexity = 20, theta = 0.5, pca = F, verbose = F, max_iter = 500, check_duplicates = F)
classes <- as.factor(dataset$Class[tsne_data])
tsne_matrix <- as.data.frame(tsne$Y)
ggplot(tsne_matrix, aes(x = V1, y = V2)) + geom_point(aes(color = classes)) + theme_minimal() + ggtitle("t-SNE visualisation of transactions") + scale_color_manual(values = c("#E69F00", "#56B4E9"))

# scaling the data using standardization and remove the first column (time) from the data set
dataset$Amount <- scale(dataset$Amount)
new_data <- dataset[, -c(1)]
head(new_data)

# split the data into training set and test set
set.seed(101)
split <- sample.split(new_data$Class, SplitRatio = 0.8)
train_data <- subset(new_data, split == TRUE)
test_data <- subset(new_data, split == FALSE)
dim(train_data)
dim(test_data)

# visualize the training data
train_data %>% ggplot(aes(x = factor(Class), y = prop.table(stat(count)), fill = factor(Class))) +
  geom_bar(position = "dodge") +
  scale_y_continuous(labels = scales::percent) +
  labs(x = 'Class', y = 'Percentage', title = 'Training Class distributions') +
  theme_grey()

#Fitting Logistic Regression Model
Logistic_Model=glm(Class~.,test_data,family=binomial())
summary(Logistic_Model)
plot(Logistic_Model)

#Plotting ROC curve to analyze its performance.
library(pROC)
logistic_predictions <- predict(Logistic_Model, test_data, type='response')
roc.curve(test_data$Class, logistic_predictions, plotit = TRUE, col = "blue")

#Fitting a Decision Tree Model
library(rpart)
library(rpart.plot)
decisionTree_model <- rpart(Class ~ . , dataset, method = 'class')
predicted_val <- predict(decisionTree_model, dataset, type = 'class')
probability <- predict(decisionTree_model, dataset, type = 'prob')
rpart.plot(decisionTree_model)

#Rose Sampling
library(ROSE)
set.seed(9560)
rose_train_data <- ROSE(Class ~ ., data = train_data)$data
table(rose_train_data$Class)

library(caret)
# Convert Class variable to factor
train_data$Class <- factor(train_data$Class)
test_data$Class <- factor(test_data$Class)

#Up Sampling
set.seed(90)
up_train_data <- upSample(x = train_data[, -30],
                          y = train_data$Class)
table(up_train_data$Class) 

#Down Sampling
set.seed(90)
down_train_data <- downSample(x = train_data[, -30],
                              y = train_data$Class)
table(down_train_data$Class) 

#Fitting a Random Forest Model
x = down_train_data[, -30]
y = down_train_data[,30]
rf_fit <- Rborist(x, y, ntree = 1000, minNode = 20, maxLeaf = 13)
rf_pred <- predict(rf_fit, test_data[,-30], ctgCensus = "prob")
prob <- rf_pred$prob
roc.curve(test_data$Class, prob[,2], plotit = TRUE, col = 'blue')

#Decision Tree on various sampling techniques
set.seed(5627)
# Build rose model
rose_fit <- rpart(Class ~ ., data = rose_train_data)

set.seed(5627)
# Build up-sampled model
up_fit <- rpart(Class ~ ., data = up_train_data)

set.seed(5627)
# Build down-sampled model
down_fit <- rpart(Class ~ ., data = down_train_data)


# Fitting a XGBoost Model
set.seed(42)
# Convert class labels from factor to numeric
labels <- as.numeric(as.character(up_train_data$Class))
labels[is.na(labels) | is.infinite(labels) | labels > 1] <- 0

# xgb fit
xgb <- xgboost(data = data.matrix(up_train_data[,-30]), 
               label = labels,
               eta = 0.1,
               gamma = 0.1,
               max_depth = 10, 
               nrounds = 300, 
               objective = "binary:logistic",
               colsample_bytree = 0.6,
               verbose = 0,
               nthread = 7
)
xgb_pred <- predict(xgb, data.matrix(test_data[,-30]))
roc.curve(test_data$Class, xgb_pred, plotit = TRUE, col = 'blue')

#Significant Variables
names <- dimnames(data.matrix(up_train_data[,-30]))[[2]]
# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = xgb)
# Nice graph
xgb.plot.importance(importance_matrix[1:10,])

#Artificial Neural Network
library(neuralnet)
ANN_model =neuralnet (Class~.,train_data,linear.output=FALSE)
plot(ANN_model)

predANN=compute(ANN_model,test_data)
resultANN=predANN$net.result
resultANN=ifelse(resultANN>0.5,1,0)


