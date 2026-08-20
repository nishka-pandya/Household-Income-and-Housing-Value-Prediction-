# --- Load Libraries ---
library(rpart)
library(rpart.plot)
library(ModelMetrics)



df <- read.csv('C:/Users/Nishka Pandya/Downloads/nup13_Data101_HW3/nup13_Data101_HW3/housing.csv')
colnames(df)
row(df)
summary(df)
head(df)


#COLUMN DESCRIPTIONS: 


#===================================================================================================================

#---------------------------------------TASK1: CREATING MACHINE LEARNING MODELS----------------------------------------------------


#--------------------------------------------------CREATING RPART MODEL-------------------------------------

#RPART CATEGORICAL TARGET VARIABLE: ***income_cat***

# Setting the seed, but as I perform cross-validation 10 times I will be changing it to draw conclusions 
set.seed(42)

# Since this data set doesn't have a lot of categorical variables, I will be creating my own. I originally tried
# To use the categorical variable ocean_proximity as a target but then I realized that it doesn't have many tuples 
# For "Island" and other unique variables, which may produce bias.

# Creating a categorical target variable focusing on income categories for households from the median_income 
# Just in context, median_income measures the median household income per row, essentially per household
df$income_cat <- cut(df$median_income,
                     breaks = c(-Inf, 3, 4.5, Inf),
                     labels = c("Low", "Mid", "High"),
                     include.lowest = TRUE)

# Want to make sure it is balanced between the three labels so the machine learning model can find patterns from 
# all three labels equally without producing bias 
table(df$income_cat)

# Taking the total rows of df 
n_tree <- nrow(df)

# Practicing the 70/30 rule where 70% is training and 30% is testing
train_id_tree <- sample(1:n_tree, 0.7 * n_tree)
train_tree <- df[train_id_tree, ]
test_tree  <- df[-train_id_tree, ]

nrow(train_tree) #14447 rows 
nrow(test_tree) #6193 rows 

# TRAIN THE MODEL 

tree <- rpart(income_cat ~ longitude + latitude + housing_median_age + total_rooms +  total_bedrooms + population +  households + median_house_value + ocean_proximity, data = train_tree)

# PLOT THE TREE 
tree
rpart.plot(tree)

# Accuracy on training data
pred_train <- predict(tree, newdata = train_tree, type="class")    
accuracy_train <- mean(train_tree$income_cat == pred_train, na.rm=TRUE)

# Accuracy on test data
pred_test <- predict(tree, newdata = test_tree, type="class")     
accuracy_test <- mean(test_tree$income_cat == pred_test, na.rm=TRUE)

cat("Accuracy on training data  :", accuracy_train, "\n")      
cat("Accuracy on testing data  :", accuracy_test, "\n")
cat("Difference between training and testing accuracy :", accuracy_train - accuracy_test, "\n")

#I will be analyzing accuracy in Task 2

#---------------------------------------------CREATING LINEAR REGRESSION MODEL---------------------------------------


#RPART NUMERICAL TARGET VARIABLE: ***median_house_value***

# Setting the seed, but as I perform cross-validation 10 times I will be changing it to draw conclusions
set.seed(42)

n_lm <- nrow(df)
train_id_lm <- sample(1:n_lm, 0.7 * n_lm)
train_lm <- df[train_id_lm, ]
test_lm  <- df[-train_id_lm, ]


# Linear Model prediction for numerical 'median_house_value'
model_lm <- lm(median_house_value ~ longitude + latitude + housing_median_age + total_rooms + total_bedrooms + population + households + median_income + ocean_proximity, data = train_lm)

# Predict and calculate MSE/RMSE as shown in slides
pred_lm <- predict(model_lm, newdata = test_lm)
mse_lm  <- mean((test_lm$median_house_value - pred_lm)^2, na.rm = TRUE)
rmse_lm <- sqrt(mse_lm)

cat("Median House Value Linear Model MSE:", mse_lm, "\n")
cat("Median House Value Linear Model RMSE:", rmse_lm, "\n")

#I will be analyzing RMSE in Task 2


#============================================================================================================================

#----------------------------------------------TASK 2: CROSS VALIDATION---------------------------------------------------

#-------------------------------RPART: ANALYSIS OF ACCURACY ON TRAINING AND TESTING DATA---------------------------------------------------------: 

#:::::::::REPORTING ACCURACY OF ONLY ONE RUN OF RPART::::::::::

#1) Training (w/ seed of 42)
cat("Accuracy on training data  :", accuracy_train, "\n")     
# The accuracy on training data is 0.6179138 (BEFORE ANY DERIVED ATTRIBUTES)
#2) Testing (we/seef of 42)
cat("Accuracy on testing data  :", accuracy_test, "\n")
# The accuracy on testing data is 0.6147263 (BEFORE ANY DERIVED ATTRIBUTES)

#The training accuracy and testing accuracy are mostly similar, with the difference of 0.002078052 (with the seed of 42). 
#However, the accuracy overall is low. It would be great to improve it a bit towards 70%+ accuracy
#Since training and testing accuracy don't have a major difference, the model isn't overfitting or underfitting.


#::::::::::NOW I WILL BE RUNNING 1-FOLD CROSS VALIDATION TEN TIMES THEN DRAW A CONCLUSION FOR RPART:::::::::::::

#RUNNING CV 1-FOLD 10 TIMES

df$income_cat <- cut(df$median_income,
                     breaks = c(-Inf, 3, 4.5, Inf),
                     labels = c("Low", "Mid", "High"),
                     include.lowest = TRUE)


# Taking the total rows of df 
n_tree <- nrow(df)

#creating a vector to store all of the values I get so I can create a boxplot to see the variance of all 
#the cv runs overall 

for (i in 1:10){
  set.seed(i)
  # Practicing the 70/30 rule where 70% is training and 30% is testing
  train_id_tree <- sample(1:n_tree, 0.7 * n_tree)
  train_tree <- df[train_id_tree, ]
  test_tree  <- df[-train_id_tree, ]
  
  nrow(train_tree) #14447 rows 
  nrow(test_tree) #6193 rows 
  
  # TRAIN THE MODEL 
  
  tree <- rpart(income_cat ~ longitude + latitude +
                  housing_median_age + total_rooms +
                  total_bedrooms + population +
                  households + median_house_value + ocean_proximity,
                data = train_tree)
  
  # PLOT THE TREE 
  tree
  rpart.plot(tree)
  
  # Accuracy on training data
  pred_train <- predict(tree, newdata = train_tree, type="class")
  accuracy_train <- mean(train_tree$income_cat == pred_train, na.rm=TRUE)
  
  # Accuracy on test data
  pred_test <- predict(tree, newdata = test_tree, type="class")
  accuracy_test <- mean(test_tree$income_cat == pred_test, na.rm=TRUE)
  
  cat("Accuracy on training data  :", accuracy_train, "\n")
  cat("Accuracy on testing data  :", accuracy_test, "\n")
  cat("Difference between training and testing accuracy :", accuracy_train - accuracy_test, "\n")
  
 
  
  
}

#!!!I manually inputted the values from seed 1-10 into a boxplot... THIS DATA ONLY REPRESENTS SEEDS 1-10!!!

#boxplot for testing data RMSE

testing_boxplot <- c(0.6119813, 0.6116583, 0.6106895, 0.6174713, 0.6024544,  0.6200549, 0.6187631 ,  0.6039076 ,  0.6192475, 0.6100436)

boxplot(testing_boxplot, main = "Accuracy from Testing Data", ylab = "Accuracy",col  = "purple")

#boxplot for training data RMSE: 

training_boxplot <- c(0.6215131 , 0.6086385, 0.6239358, 0.6213747, 0.6203364, 0.6253201, 0.613553, 0.6083616,  0.6221361, 0.6180522 )

boxplot(training_boxplot, main = "Accuracy from Training Data", ylab = "Accuracy", col  = "lightpink")

#boxplot w/ both training and testing data, colored 

boxplot(training_boxplot, testing_boxplot, names = c("Training", "Testing"),  ylab = "Accuracy",  col  = c("lightpink", "purple"))


#:::::::::::::::::::CONCLUSION-- RPART:::::::::::::::::::

#After running ten times with seeds from 1-10 I noticed that overall there is very little difference between training and testing data. 
#So, the model isn't overfitting or underfitting as seen by the boxplots of training and testing data; they are overlapping.
#The accuracy for the training and testing data could however be better as it is pretty low. Boosting it to 
#70%+ would be good.



#------------------------------------LINEAR REGRESSION: ANALYSIS OF MSE ON TRAINING AND TESTING DATA------------------------------------------:

#:::::::::REPORTING MSE OF ONLY ONE RUN OF LM::::::::::


#1) Testing (w/ seed of 42)
cat("Median House Value Linear Model RMSE:", rmse_lm, "\n")
# The RMSE on testing data is 67533.97 (BEFORE ANY DERIVED ATTRIBUTES)

#With the seed of 42, the RMSE on testing data is 67533.97.I will now run Cross Validaton 10 times.

#RUNNING CV 1-FOLD 10 TIMES 

for (i in 1:10){
  set.seed(i)
  
  n_lm <- nrow(df)
  train_id_lm <- sample(1:n_lm, 0.7 * n_lm)
  train_lm <- df[train_id_lm, ]
  test_lm  <- df[-train_id_lm, ]
  
  
  # Linear Model prediction for numerical 'median_house_value'
  model_lm <- lm(median_house_value ~ longitude + latitude +  housing_median_age + total_rooms +  total_bedrooms + population + households + median_income + ocean_proximity, data = train_lm)
  
  # Predict and calculate MSE/RMSE as shown in slides
  pred_lm <- predict(model_lm, newdata = test_lm)
  mse_lm  <- mean((test_lm$median_house_value - pred_lm)^2, na.rm = TRUE)
  rmse_lm <- sqrt(mse_lm)
  
  pred_lm_train <- predict(model_lm, newdata = train_lm)
  mse_lm_train <- mean((train_lm$median_house_value - pred_lm_train)^2, na.rm = TRUE)
  rmse_lm_train <- sqrt(mse_lm_train)
  
  #cat("Median House Value Linear Model MSE for TESTING DATA:", mse_lm, "\n")
  cat("Median House Value Linear Model RMSE for TESTING DATA:", rmse_lm, "\n")
  cat("\n")
  cat("Median House Value Linear Model RMSE for TRAINING DATA:", rmse_lm_train, "\n")
  
 
  
  
}

#!!!I manually inputted the values from seed 1-10 into a boxplot... THIS DATA ONLY REPRESENTS SEEDS 1-10!!!

#boxplot for testing data RMSE

testing_boxplot <- c(68474.49, 69295.95, 69188.28, 70313.28, 68394.18, 68567.85, 68612.49, 69404.8, 68394.04, 70458.22)

boxplot(testing_boxplot,
        main = "RMSE from Testing Data",
        ylab = "RMSE",
        col  = "purple")

#boxplot for training data RMSE: 

training_boxplot <- c(68723.21, 68382.35, 68450.46, 67954.43, 68757.57, 68707.36, 68706.11, 68318.67, 68768.26, 67852.24)

boxplot(training_boxplot,
        main = "RMSE from Training Data",
        ylab = "RMSE",
        col  = "lightpink")

#boxplot w/ both training and testing data, colored 

boxplot(training_boxplot, testing_boxplot,
        names = c("Training", "Testing"),
        ylab = "RMSE",
        col  = c("lightpink", "purple"))

#CONCLUSION-- LM: 

#Overall the RMSE is around 68,000-70,000. To overall measure how good the RMSE is, one of the methods
#I chose to do was compare it to the mean house value. 

mean_of_house_value <- mean(df$median_house_value)
69000/mean_of_house_value

#The RMSE accounts for 33% of the mean_of_house_value. Overall, the machine learning model performs well, but 
#there is definitely room for improvement. And as it can be seen by the boxplots earlier, the training and testing data overlap
#a bit, so there is not too much overfitting or underfitting.



#============================================================================================================================

#----------------------------------------------TASK 3: DERIVED ATTRIBUTE---------------------------------------------------


#--------------------------------------------ADDING DERIVED ATTRIBUTE TO R PART------------------------------------------

#row-based attribute: 

set.seed(1)

#ROW BASED ATTRIBUTE:
df$roomsByHome <- df$total_rooms / df$households

#I chose this row based attributebecause I was thinking that one of the ways we can improve rpart accuracy is by figuring out the amount of rooms each
#home has. I was thinking that since more costly/rich houses have more rooms while less costly have less. I thought this may 
#allow the model to identify more differentiating patterns.

# Taking the total rows of df 
n_tree <- nrow(df)

# Practicing the 70/30 rule where 70% is training and 30% is testing
train_id_tree <- sample(1:n_tree, 0.7 * n_tree)
train_tree <- df[train_id_tree, ]
test_tree  <- df[-train_id_tree, ]

nrow(train_tree) #14447 rows 
nrow(test_tree) #6193 rows 

# TRAIN THE MODEL 

tree <- rpart(income_cat ~ longitude + latitude + housing_median_age + total_rooms + total_bedrooms + population + households + median_house_value + ocean_proximity + roomByHome, data = train_tree)

# PLOT THE TREE 
tree
rpart.plot(tree)

# Accuracy on training data
pred_train <- predict(tree, newdata = train_tree, type="class")
accuracy_train <- mean(train_tree$income_cat == pred_train, na.rm=TRUE)

# Accuracy on test data
pred_test <- predict(tree, newdata = test_tree, type="class")
accuracy_test <- mean(test_tree$income_cat == pred_test, na.rm=TRUE)

cat("Accuracy on training data  :", accuracy_train, "\n")
cat("Accuracy on testing data  :", accuracy_test, "\n")
cat("Difference between training and testing accuracy :", accuracy_train - accuracy_test, "\n")


# RUNNING 1-FOLD CROSS VALIDATION ON THE ROW DERIVED ATTRIBUTE MODEL TEN TIMES AND COMPARING TO ORIGINAL MODEL BEFORE
# THE DERIVED ATTRIBUTE 

df$roomByHome <- df$total_rooms / df$households

for (i in 1:10){
  set.seed(i)
  # Practicing the 70/30 rule where 70% is training and 30% is testing
  train_id_tree <- sample(1:n_tree, 0.7 * n_tree)
  train_tree <- df[train_id_tree, ]
  test_tree  <- df[-train_id_tree, ]
  
  nrow(train_tree) #14447 rows 
  nrow(test_tree) #6193 rows 
  
  # TRAIN THE MODEL 
  
  #tree <- rpart(income_cat ~ longitude + latitude +
                #  housing_median_age + total_rooms +
                #  total_bedrooms + population +
                #  households + median_house_value + ocean_proximity + roomByHome,
               # data = train_tree)
  
  tree <- rpart(income_cat ~ longitude + latitude +
                  housing_median_age + population +
                  households + median_house_value + ocean_proximity + roomByHome,
                data = train_tree)
  
  # PLOT THE TREE 
  tree
  rpart.plot(tree)
  
  # Accuracy on training data
  pred_train <- predict(tree, newdata = train_tree, type="class")
  accuracy_train <- mean(train_tree$income_cat == pred_train, na.rm=TRUE)
  
  # Accuracy on test data
  pred_test <- predict(tree, newdata = test_tree, type="class")
  accuracy_test <- mean(test_tree$income_cat == pred_test, na.rm=TRUE)
  
  cat("Accuracy on training data  :", accuracy_train, "\n")
  cat("Accuracy on testing data  :", accuracy_test, "\n")
  cat("Difference between training and testing accuracy :", accuracy_train - accuracy_test, "\n")
  
}

#CONCLUSION-- COMPARING ORIGINAL RPART WITH THE RPART WITH ROW BASED DERIVED ATTRIBUTE

#comparing testing accuracy from original with testing accuracy after row based derived attribute 

derived_testing_data <- c(0.6969159, 0.687066, 0.6957856, 0.6849669, 0.6836751, 0.6927176, 0.687066, 0.6890037, 0.6956241, 0.6994994)

testing_boxplot <- c(0.6119813, 0.6116583, 0.6106895, 0.6174713, 0.6024544,  0.6200549, 0.6187631 ,  0.6039076 ,  0.6192475, 0.6100436)


boxplot(derived_testing_data, testing_boxplot,
        names = c("Derived", "Original"),
        ylab = "Accuracy",
        col  = c("lightpink", "purple"))

#Based on these boxplots, there is a significant increase in accuracy of the testing data after adding the row-based
#derived attribute. Therefore, this improvement statistically significant.



#--------------------------------------------ADDING DERIVED ATTRIBUTE TO LM--------------------------------------------


#for the aggregated based derived attribute, I decided to create two attributes.
#I wanted to create a group of all households that share the same location to be able to 
#calculate median_house_value, so in this case  I compared through average of longitude and lattitude 


unique(df$latitude)
unique(df$longitude)


#create a group of up to ten different latitude and longitude ranges
df$agg_latitude <- cut(df$latitude,  breaks = 10)
df$agg_longitude <- cut(df$longitude, breaks = 10)

unique(df$agg_latitude)
unique(df$agg_longitude)

#find average median house value per each group of long and lat
mean_latitude <- tapply(df$median_house_value, df$agg_latitude, mean)
mean_longitude <- tapply(df$median_house_value, df$agg_longitude, mean)

#create a new column of the average long and lat groups
df$mean_latitude <- mean_latitude[as.character(df$agg_latitude)]
df$mean_longitude <- mean_longitude[as.character(df$agg_longitude)]

unique(df$mean_latitude)
unique(df$mean_longitude)

# RUNNING 1-FOLD CROSS VALIDATION ON THE ROW DERIVED ATTRIBUTE MODEL TEN TIMES AND COMPARING TO ORIGINAL MODEL BEFORE
# THE DERIVED ATTRIBUTE 

for (i in 1:10){
  set.seed(i)
  n_lm <- nrow(df)
  train_id_lm <- sample(1:n_lm, 0.7 * n_lm)
  train_lm <- df[train_id_lm, ]
  test_lm  <- df[-train_id_lm, ]
  
  # Linear Model prediction for numerical 'price'
  lm_base <- lm(median_house_value ~ longitude + latitude +
                  housing_median_age + total_rooms +
                  total_bedrooms + population +
                  households + median_income + ocean_proximity + mean_latitude + mean_longitude,
                data = train_lm)
  
  # Predict and calculate MSE/RMSE as shown in slides
  pred_lm_train <- predict(lm_base, newdata = train_lm)
  
  pred_lm_test <- predict(lm_base, newdata = test_lm)
  mse_lm  <- mean((test_lm$median_house_value - pred_lm_test)^2, na.rm = TRUE)
  rmse_lm <- sqrt(mse_lm)
  
  cat("Salary Linear Model MSE:", mse_lm, "\n")
  cat("Manual Salary Linear Model RMSE:", rmse_lm, "\n")
  
}

#!!BASED ONLY ON SEEDS FROM 1-10!!

testing_boxplot <- c(68474.49, 69295.95, 69188.28, 70313.28, 68394.18, 68567.85, 68612.49, 69404.8, 68394.04, 70458.22)

derived_testing_boxplot <- c(67183.53, 67884.54, 67977.12, 69214.12, 67184.27, 67505.22, 67438.91, 68106.03, 67084.28, 69102.18)


boxplot(derived_testing_boxplot, testing_boxplot,
        names = c("Derived", "Original"),
        ylab = "RMSE",
        col  = c("lightpink", "purple"))


#CONCLUSION:

#Based on the boxplots between the RMSE of the original model and the model after adding the aggregated derived attribute
#the derived attribute's RMSE decreased slightly, meaning it improved a bit. I think it improved because overall,
#having a smaller RMSE range is better. Therefore, although the improvement is small, it is
#statistically significant because the boxplots are not overlapping. 
































# Linear Model prediction for numerical 'price'
lm_base <- lm(median_house_value ~ longitude + latitude +
                housing_median_age + total_rooms +
                total_bedrooms + population +
                households + median_income + ocean_proximity,
              data = train_lm)

# Predict and calculate MSE/RMSE as shown in slides
pred_lm_train <- predict(lm_base, newdata = train_lm)

pred_lm_test <- predict(lm_base, newdata = test_lm)
mse_lm  <- mean((test_lm$median_house_value - pred_lm_test)^2, na.rm = TRUE)
rmse_lm <- sqrt(mse_lm)

cat("Salary Linear Model MSE:", mse_lm, "\n")
cat("Manual Salary Linear Model RMSE:", rmse_lm, "\n")

cat("Mean house value  :", round(mean(df$median_house_value), 0), "\n")
cat("Your RMSE         :", rmse_lm, "\n")
cat("RMSE as % of mean :", round(rmse_lm / mean(df$median_house_value) * 100, 1), "%\n")

cor(df$total_rooms    / df$households, df$median_income, use = "complete.obs")
cor(df$total_bedrooms / df$households, df$median_income, use = "complete.obs")
cor(df$population     / df$households, df$median_income, use = "complete.obs")

df$people_per_room <- df$total_rooms / df$households

num_cols <- sapply(df, is.numeric)
round(cor(df[, num_cols], use = "complete.obs"), 2)



df$people_per_room <- df$total_rooms / df$households

#-----LM----- 
set.seed(42)

n_lm <- nrow(df)
train_id_lm <- sample(1:n_lm, 0.7 * n_lm)
train_lm <- df[train_id_lm, ]
test_lm  <- df[-train_id_lm, ]

# Linear Model prediction for numerical 'price'
lm_base <- lm(median_house_value ~ longitude + latitude +
                housing_median_age + total_rooms +
                total_bedrooms + population +
                households + median_income + ocean_proximity,
              data = train_lm)

# Predict and calculate MSE/RMSE as shown in slides
pred_lm_train <- predict(lm_base, newdata = train_lm)

pred_lm_test <- predict(lm_base, newdata = test_lm)
mse_lm  <- mean((test_lm$median_house_value - pred_lm_test)^2, na.rm = TRUE)
rmse_lm <- sqrt(mse_lm)

cat("Salary Linear Model MSE:", mse_lm, "\n")
cat("Manual Salary Linear Model RMSE:", rmse_lm, "\n")

cat("Mean house value  :", round(mean(df$median_house_value), 0), "\n")
cat("Your RMSE         :", rmse_lm, "\n")
cat("RMSE as % of mean :", round(rmse_lm / mean(df$median_house_value) * 100, 1), "%\n")


df$lat_bucket <- round(df$latitude, 0)
lat_avg <- tapply(df$median_house_value, df$lat_bucket, mean)
df$lat_mean_value <- lat_avg[as.character(df$lat_bucket)]

df$lon_bucket <- round(df$longitude, 0)
lon_avg <- tapply(df$median_house_value, df$lon_bucket, mean)
df$lon_mean_value <- lon_avg[as.character(df$lon_bucket)]

df$geo_bucket <- paste(round(df$latitude, 0), round(df$longitude, 0))
geo_avg <- tapply(df$median_house_value, df$geo_bucket, mean)
df$geo_mean_value <- geo_avg[as.character(df$geo_bucket)]

df$age_bucket <- cut(df$housing_median_age, breaks = c(0, 10, 20, 30, 40, Inf))
age_avg <- tapply(df$median_house_value, df$age_bucket, mean)
df$age_mean_value <- age_avg[as.character(df$age_bucket)]

cor(df$lat_mean_value, df$median_house_value, use = "complete.obs")
cor(df$lon_mean_value, df$median_house_value, use = "complete.obs")
cor(df$geo_mean_value, df$median_house_value, use = "complete.obs")
cor(df$age_mean_value, df$median_house_value, use = "complete.obs")

set.seed(i)

n_lm <- nrow(df)
train_id_lm <- sample(1:n_lm, 0.7 * n_lm)
train_lm <- df[train_id_lm, ]
test_lm  <- df[-train_id_lm, ]

# Linear Model prediction for numerical 'price'
lm_base <- lm(median_house_value ~ longitude + latitude +
                housing_median_age + total_rooms +
                total_bedrooms + population +
                households + median_income + ocean_proximity + geo_mean_value,
              data = train_lm)

# Predict and calculate MSE/RMSE as shown in slides
pred_lm_train <- predict(lm_base, newdata = train_lm)

pred_lm_test <- predict(lm_base, newdata = test_lm)
mse_lm  <- mean((test_lm$median_house_value - pred_lm_test)^2, na.rm = TRUE)
rmse_lm <- sqrt(mse_lm)

cat("Salary Linear Model MSE:", mse_lm, "\n")
cat("Manual Salary Linear Model RMSE:", rmse_lm, "\n")

cat("Mean house value  :", round(mean(df$median_house_value), 0), "\n")
cat("Your RMSE         :", rmse_lm, "\n")
cat("RMSE as % of mean :", round(rmse_lm / mean(df$median_house_value) * 100, 1), "%\n")


