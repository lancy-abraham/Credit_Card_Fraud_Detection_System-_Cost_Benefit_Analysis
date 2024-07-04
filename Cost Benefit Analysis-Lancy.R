library(tidyverse)
library(dplyr)
library(caret)
library(MASS)
library(e1071)
library(class)
library(xgboost)
library(ROSE)
library(dplyr)

options(warn=-1)

# Reading the test dataset
fraud_test <- read.csv("C:/Users/Lancy/Desktop/IIMK/4. DS Strategy and Capstone/4. Module 1 - Capstone Project - Credit Card Fraud Detection/Project Details/Credit Card Fraud Analysis & Cost Benefit Analysis- Lancy/fraudTest.csv")
head(fraud_test)

# Reading the train dataset
fraud_train <- read.csv("C:/Users/Lancy/Desktop/IIMK/4. DS Strategy and Capstone/4. Module 1 - Capstone Project - Credit Card Fraud Detection/Project Details/Credit Card Fraud Analysis & Cost Benefit Analysis- Lancy/fraudTrain.csv")
head(fraud_train)

# Converting columns to datetime for fraudTest
fraud_test$dob <- as.POSIXct(fraud_test$dob)
fraud_test$trans_date_trans_time <- as.POSIXct(fraud_test$trans_date_trans_time)

fraud_train$dob <- as.POSIXct(fraud_train$dob)
fraud_train$trans_date_trans_time <- as.POSIXct(fraud_train$trans_date_trans_time)

# Creating a new column called Transaction date and converting into datetime
# For fraud_test dataframe
fraud_test$Transaction_Date <- as.POSIXct(fraud_test$trans_date_trans_time, format="%m/%d/%Y %H:%M")
# For fraud_train dataframe
fraud_train$Transaction_Date <- as.POSIXct(fraud_train$trans_date_trans_time, format="%m/%d/%Y %H:%M")

# Creating a new column called Age
# For fraud_test dataframe
fraud_test$Age <- as.integer(round((as.numeric(difftime(fraud_test$Transaction_Date, fraud_test$dob, units = "days")))/365))

# For fraud_train dataframe
fraud_train$Age <- as.integer(round((as.numeric(difftime(fraud_train$Transaction_Date, fraud_train$dob, units = "days")))/365))

# Creating a new column called Transaction Time
# For fraud_test dataframe
fraud_test$Transaction_Time <- format(as.POSIXct(fraud_test$trans_date_trans_time, format = "%Y:%m:%d %H:%M:%S"), "%H:%M:%S")

# For fraud_train dataframe
fraud_train$Transaction_Time <- format(as.POSIXct(fraud_train$trans_date_trans_time, format = "%Y:%m:%d %H:%M:%S"), "%H:%M:%S")


# Creating a function to calculate the Day of Week
# Define DoW function
DoW <- function(x) {
  days <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
  return(days[as.integer(format(x, "%u"))])
}

# For fraud_test dataframe
fraud_test$Day_of_Week <- sapply(fraud_test$Transaction_Date, DoW)

# For fraud_train dataframe
fraud_train$Day_of_Week <- sapply(fraud_train$Transaction_Date, DoW)

# Creating a new column Month,Making Gender column binary and Ensuring Day of Week is represented numerically
# For fraud_test dataframe
fraud_test$Month <- as.integer(format(as.Date(fraud_test$trans_date_trans_time, "%Y:%m:%d"), "%m"))
fraud_test$gender <- ifelse(fraud_test$gender == "F", 1, 0)
fraud_test$Day_of_Week <- match(fraud_test$Day_of_Week, c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))

# For fraud_train dataframe
fraud_train$Month <- as.integer(format(as.Date(fraud_train$trans_date_trans_time, "%Y:%m:%d"), "%m"))
fraud_train$gender <- ifelse(fraud_train$gender == "F", 1, 0)
fraud_train$Day_of_Week <- match(fraud_train$Day_of_Week, c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))

# Creating a function to calculate the distance between customer's base location and merchant location
# Define a function to convert degrees to radians
deg2rad <- function(deg) {
  return (deg * pi / 180)
}

# Define a function to convert radians to degrees
rad2deg <- function(rad) {
  return (rad * 180 / pi)
}


library(geosphere)

haversineDistance <- function(lat1, lon1, lat2, lon2) {
  d <- distm(c(lon1, lat1), c(lon2, lat2), fun = distHaversine) / 1000  # convert to km
  return(d)
}

Dist <- apply(fraud_train[, c("lat", "long", "merch_lat", "merch_long")], 1, function(x) {
  haversineDistance(x[1], x[2], x[3], x[4])
})

fraud_train$Dist <- Dist

# Calling the function created earlier to calculate the Distance between the customer's home location and the location of transactions (fraudulent or otherwise)

Dist2 <- apply(fraud_test[, c("lat", "long", "merch_lat", "merch_long")], 1, function(x) {
  haversineDistance(x[1], x[2], x[3], x[4])
})

# Adding the calculated Distance column in the test dataframe
fraud_test$Dist <- Dist2
head(fraud_test)

library(dplyr)

fraud_test <- fraud_test %>% 
  mutate(
    amt = log(amt),
    city_pop = log(city_pop)
  )

fraud_train <- fraud_train %>% 
  mutate(
    amt = log(amt),
    city_pop = log(city_pop)
  )

#scatterplot to visualize the relationship between the amt and is_fraud variables in the fraudTrain dataset

ggplot(fraud_train[fraud_train$is_fraud == 1, ], aes(x = amt, y = is_fraud)) +
  geom_point() +
  labs(title = "Relationship between the amt and is_fraud variables") +
  theme_minimal()

#scatterplot to visualize the relationship between the amt and city_pop variables in the fraudTrain dataset
ggplot(fraud_train[fraud_train$is_fraud == 1, ], aes(x = amt, y = city_pop)) +
  geom_point(color = "blue") +
  labs(title = "Relationship between the amt and city_pop variables") +
  theme_minimal()

#distribution plot to visualize the distribution of the amt variable in the fraudTrain dataset

ggplot(fraud_train, aes(x = amt)) +
  geom_histogram(bins = 10, color = "blue", fill = "grey", alpha = 0.5) +
  geom_density() +
  labs(title = "Univariate Analysis - Amount") +
  theme_minimal()

#distribution plot to visualize the distribution of the city_pop variable in the fraudTrain dataset

ggplot(fraud_train, aes(x = city_pop)) +
  geom_histogram(bins = 10, color = "red", fill = "grey", alpha = 0.5) +
  geom_density() +
  labs(title = "Univariate Analysis - City Population") +
  theme_minimal()

#distribution of values in the 'Age' column of the 'fraudTrain' dataset

ggplot(fraud_train, aes(x = Age)) +
  geom_histogram(bins = 10, color = "green", fill = "grey", alpha = 0.5) +
  geom_density() +
  labs(title = "Univariate Analysis - Age") +
  theme_minimal()


#distribution of values in the 'Dist' column of the 'fraudTrain' dataset

ggplot(fraud_train, aes(x = Dist)) +
  geom_histogram(bins = 10, color = "black", fill = "grey", alpha = 0.5) +
  geom_density() +
  labs(title = "Univariate Analysis - Distance") +
  theme_minimal()

# barplot to visualize the distribution of the target variable
x <- c(0, 1)
y <- c(sum(fraud_train$is_fraud == 0), sum(fraud_train$is_fraud == 1))

barplot(y, names.arg = x, col = "steelblue", xlab = "is_fraud", ylab = "Count", main = "Univariate Analysis - Target Variable Distribution")

#barplot to visualize the distribution of the gender variable in the fraudTrain dataset.
x <- c('F', 'M')
y <- c(fraud_train$gender %>% table() %>% as.vector())

ggplot(data = data.frame(x, y), aes(x = x, y = y)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Univariate Analysis - Gender Distribution")

# Checking Class Imbalance
classes_train <- table(fraud_train$is_fraud)
normal_share_train <- classes_train[1]/nrow(fraud_train)*100
fraud_share_train <- classes_train[2]/nrow(fraud_train)*100

classes_test <- table(fraud_test$is_fraud)
normal_share_test <- classes_test[1]/nrow(fraud_test)*100
fraud_share_test <- classes_test[2]/nrow(fraud_test)*100

par(mfrow=c(2,1), mar=c(4, 4, 2, 1))
barplot(classes_train, names.arg=c("Non-Fraud", "Fraud"), col=c("yellow", "blue"), main="Train", ylab="Number of transactions")
text(0.2, 0.5, paste0(format(normal_share_train, digits=4), "%"), cex=1.2)
text(0.7, 0.5, paste0(format(fraud_share_train, digits=4), "%"), cex=1.2)

barplot(classes_test, names.arg=c("Non-Fraud", "Fraud"), col=c("yellow", "blue"), main="Test", ylab="Number of transactions")
text(0.2, 0.5, paste0(format(normal_share_test, digits=4), "%"), cex=1.2)
text(0.7, 0.5, paste0(format(fraud_share_test, digits=4), "%"), cex=1.2)

#unique list of cities with fraud cases
fraud_city <- data.frame(table(fraud_train$city, fraud_train$is_fraud))
fraud_city <- fraud_city[fraud_city$Freq > 0 & fraud_city$Var2 == 1,]
print(unique(fraud_city$Var1))

#unique list of states with fraud cases
fraud_state <- data.frame(table(fraud_train$state, fraud_train$is_fraud))
fraud_state <- fraud_state[fraud_state$Freq > 0 & fraud_state$Var2 == 1,]
print(unique(fraud_state$Var1))

#unique list of states with fraud jobs.
fraud_job <- data.frame(table(fraud_train$job, fraud_train$is_fraud)[,1:2])
fraud_job <- subset(fraud_job, fraud_job$Freq > 0 & fraud_job$Var2 == 1)$Var1
print(fraud_job)

#Finding distance from customer location to merchant location in degrees latitude and degrees longitude
fraud_train$lat_dist <- abs(round(fraud_train$merch_lat-fraud_train$lat,3))
fraud_train$long_dist <- abs(round(fraud_train$merch_long-fraud_train$long,3))

fraud_test$lat_dist <- abs(round(fraud_test$merch_lat-fraud_test$lat,3))
fraud_test$long_dist <- abs(round(fraud_test$merch_long-fraud_test$long,3))

head(fraud_train[, c('merch_lat', 'lat', 'lat_dist', 'merch_long', 'long', 'long_dist')])


#the number of unique values in each column of fraud_train.
library(dplyr)
sapply(fraud_train, n_distinct)

sapply(fraud_test, function(x) length(unique(x)))
sapply(fraud_train, function(x) length(unique(x)))

#calculating fraud percentage based on category
round(prop.table(table(fraud_train$category))*100, 2)

#the percentage distribution of the 'gender' column
round(prop.table(table(fraud_train$gender))*100, 2)

#calculate the percentage of occurrences for each unique value in the "city" column 
round(prop.table(table(fraud_train$city))*100, 2)

#calculate the percentage distribution of each unique value in the "state" column
round(prop.table(table(fraud_train$state)) * 100, 2)

#percentage of unique values in the 'job' column
round(prop.table(table(fraud_train$job))*100, 2)

#Dropping variables not useful for model building
drop_cols <- c('Unnamed: 0','trans_date_trans_time','city','lat','long','job','dob','merch_lat','merch_long','Transaction_Date','Transaction_Time','cc_num','merchant','first','last','street','zip','trans_num','unix_time')

fraud_train1 <- fraud_train[, !(names(fraud_train) %in% drop_cols)]
fraud_test1 <- fraud_test[, !(names(fraud_test) %in% drop_cols)]

head(fraud_test1)

# Creating dummy variables for Category
fraud_train2 <- model.matrix(~ category + 0, data = fraud_train1)
fraud_train2 <- as.data.frame(fraud_train2)
colnames(fraud_train2) <- gsub("category", "", colnames(fraud_train2))

fraud_test2 <- model.matrix(~ category + 0, data = fraud_test1)
fraud_test2 <- as.data.frame(fraud_test2)
colnames(fraud_test2) <- gsub("category", "", colnames(fraud_test2))

# Dropping State column
fraud_train3 <- fraud_train2[, !colnames(fraud_train2) %in% c("state")]
fraud_test3 <- fraud_test2[, !colnames(fraud_test2) %in% c("state")]


library(randomForest)
library(caret)

# Splitting train and test dataset into X and y
X_train <- fraud_train[, !(names(fraud_train) %in% c("trans_date_trans_time", "amt", "trans_num", "is_fraud"))]
y_train <- fraud_train$is_fraud

X_test <- fraud_test[, !(names(fraud_test) %in% c("trans_date_trans_time", "amt", "trans_num", "is_fraud"))]
y_test <- fraud_test$is_fraud

# Train a random forest model
library(ranger)
rf <- ranger(y = y_train, x = X_train, num.trees = 100, mtry = 5, importance = "impurity")

#predicted values for the target variable using the trained random forest model
y_train_pred <- predict(rf, X_train)
head(y_train_pred)
#predicted values for the target variable using the trained random forest model
y_test_pred <- predict(rf, X_test)
head(y_test_pred)
 
#classification report for the training set predictions
library(caret)

 # Convert variables to factors with the same levels
 y_train_pred <- factor(y_train_pred, levels = levels(factor(y_train)))
 y_test_pred <- factor(y_test_pred, levels = levels(factor(y_test)))
 y_train <- factor(y_train, levels = levels(factor(y_train)))
 y_test <- factor(y_test, levels = levels(factor(y_test)))
 
 y_pred=merge.data.frame(y_train,y_test)
 
 # Subset of "fraud_test"
 fraud_test_merge <- fraud_test[, c("trans_date_trans_time", "amt", "trans_num", "is_fraud")]
 
 # Create data frame of predicted values for test set
 fraud_test_pred <- data.frame(y_test_pred)
 
 # Make sure the number of rows in the two data frames match
 n <- nrow(fraud_test_merge)
 fraud_test_pred <- fraud_test_pred[1:n, ]
 
 # Merge the two data frames
 fraud_test_final <- cbind(fraud_test_merge, fraud_test_pred)
 
 # Rename column
 colnames(fraud_test_final)[ncol(fraud_test_final)] <- "is_fraud_pred"
 
 # Group rows by "is_fraud"
 table(fraud_test_final$is_fraud)
 
 # Group rows by "is_fraud_pred"
 table(fraud_test_final$is_fraud_pred)
 
# Rename column
colnames(fraud_test_final)[ncol(fraud_test_final)] <- "is_fraud_pred"

# Count the number of occurrences of each unique value in "is_fraud"
table(fraud_test_final$is_fraud)

# Group rows by "is_fraud_pred"
table(fraud_test_final$is_fraud_pred)

# Merge train and test datasets
fraud_merge_final <- rbind(fraud_train_final, fraud_test_final)

# Shape of the data frame
dim(fraud_merge_final)

#Creating month and year columns
fraud_merge_final$month <- format(as.Date(fraud_merge_final$trans_date_trans_time, format = "%Y-%m-%d %H:%M:%S"), "%m")
fraud_merge_final$year <- format(as.Date(fraud_merge_final$trans_date_trans_time, format = "%Y-%m-%d %H:%M:%S"), "%Y")
head(fraud_merge_final)

#Group by variable creation
g <- group_by(fraud_merge_final, year, month)

avg_fraud_amt <- sum(fraud_merge_final[fraud_merge_final$is_fraud == 1,]$amt)
print(avg_fraud_amt)

#Average number of transactions per month
nrow(fraud_merge_final)/24

#Average Number of fraudulent transactions per month
avg_fraudtrans_pm <- nrow(fraud_merge_final[fraud_merge_final$is_fraud == 1,])/24
print(avg_fraudtrans_pm)

fraud <- rbind(fraud_test, fraud_train)
head(fraud)

#Average amount per fraud transaction
avg_fraud_amt <- mean(fraud_merge_final[fraud_merge_final$is_fraud == 1,]$amt)*95.54
print(avg_fraud_amt)

#Calculating the cost incurred before deploying the model based on the first point descibed above:
cost_before_model <- avg_fraud_amt*avg_fraudtrans_pm
cost_before_model

#Average number of transactions per month detected as fraud by the model
nrow(fraud_merge_final[fraud_merge_final$is_fraud_pred == 1,])/24

#Average number of transactions per month that are fraudulent but are not detected by the model
Undetected_frauds <- fraud_merge_final[fraud_merge_final$is_fraud_pred==0 & fraud_merge_final$is_fraud==1,]
Non_detected <- nrow(Undetected_frauds)/24*0.18426
print(Non_detected)

#Calculating the cost incurred after deploying the model based on the first point descibed above:
cost_after_model <- Non_detected*avg_fraud_amt
cost_after_model
