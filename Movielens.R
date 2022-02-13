##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#-----------------------------------------------------------------------------------------------------------
# 1. Exploring the Data
#-----------------------------------------------------------------------------------------------------------
# Number of distinct users and movies
edx %>% summarize(Users = n_distinct(userId),
                  Movies = n_distinct(movieId))

#Checking for Number of missing values, if any
sapply(edx, function(x) sum(is.na(x)))

#Viewing data
head(edx)

#Plotting Distribution of Ratings
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.20, color = "black") +
  ggtitle("Distribution of ratings")

# Number of Ratings per Movie
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 50, color = "black") +
  scale_x_log10() +
  xlab("Number of ratings") +
  ylab("Number of movies") +
  ggtitle("Number of Ratings per Movie")


# Making a plot which shows the positive correlation of #ratings a movie receives and its avg rating
edx %>% group_by(movieId) %>% summarise(n = n(), mean_rate = mean(rating)) %>% 
  arrange(desc(n)) %>% 
  ggplot(aes(mean_rate,n)) + geom_point()+ scale_y_continuous(trans = "log2")+ 
  geom_smooth(method = lm) + ylab("#Ratings per movie") + 
  xlab("Mean rating per movie") + ggtitle("#Ratings vs. Mean Rating")


# Age Model: Using b_a to represent ratings on movies with regards to age
age_effect<- edx1 %>% 
  group_by(age_at_rating) %>%
  summarize(b_a = mean(rating)-mu)
age_effect %>% qplot(b_a, geom ="histogram", bins = 10, data = ., color = I("black"), fill = "black")

# Year Released Model: Using b_ry to represent ratings on movies with regards to Year released
Year_released_effect<- edx1 %>% 
  group_by(year_released) %>%
  summarize(b_ry = mean(rating)-mu)
Year_released_effect %>% qplot(b_ry, geom ="histogram", bins = 10, data = ., color = I("black"), fill = "black")

# User Model: Using b_u to represent ratings on movies with regards to user
User_effect<- edx1 %>% 
  group_by(userId) %>%
  summarize(b_u = mean(rating)-mu)
User_effect %>% qplot(b_u, geom ="histogram", bins = 10, data = ., color = I("black"), fill = "black")

# Movie Model: Using b_i to represent ratings on movies with regards to a certain movie 
Movie_effect<- edx1 %>% 
  group_by(movieId) %>%
  summarize(b_i = mean(rating)-mu)
Movie_effect %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"), fill = "black")


# -------------------
# Feature Engineering
# -------------------

# edx set now edx1 after timestamp conversion, title split and genres separation
# partition edx1 into train and test sets
# The validation set will only be used for final validation.
# loading lubridate library to work on the timestamp 

if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

# Convert timestamp to datetime format called year_rated
edx1 <- edx %>% mutate(year_rated = year(as_datetime(timestamp)))

# Separate the release year of the movie from title in the edx1 dataset
# So timestamp column is replaced with year_rated and age at rating 
# while title becomes both title and year_released
edx1 <- edx1 %>% mutate(title = str_replace(title,"^(.+)\\s\\((\\d{4})\\)$","\\1__\\2" )) %>%
separate(title,c("title","year_released"),"__") %>% select(-timestamp)
edx1 <- edx1 %>% mutate(age_at_rating= as.numeric(year_rated)-as.numeric(year_released))


# edx1 becomes edx_final in which the mixture of genres is split into different rows
genres_edx <- edx1 %>% separate_rows(genres,sep = "\\|") %>% mutate(value=1)
n_distinct(genres_edx$genres)  # 20: there are 20 different types of genres
final_genres <- genres_edx %>% group_by(genres) %>% summarize(n=n())
final_genres

# edx_final dataset is split into train (for training the model) and test (for testing the model) 
test_index <- createDataPartition(
y = edx1$rating, times = 1, p = 0.1, list = FALSE)
train <- edx1[-test_index,]
temp <- edx1[test_index,]

# Make sure userId and movieId in test set are also in train set
test <- temp %>%
semi_join(train, by = "movieId") %>%
semi_join(train, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test)
train <- rbind(train, removed, temp)

# Remove unnecessary variables
rm(test_index, temp, removed)

# RMSE is Residual Mean Squared Error
RMSE <- function(true_ratings, predicted_ratings){
sqrt(mean((true_ratings - predicted_ratings)^2))
}

# -------------------
# Modelling Features
# -------------------

# Mean Ratings for all Movies irrespective of the the User (naive mean)
mu <- mean(train$rating)
rmse_naivemean <- RMSE(test$rating, mu)
model_scores <- data.frame(method='Naive means',
rmse=rmse_naivemean)
 

# Effects of Movie bias on rating
b_i <- train %>%
group_by(movieId) %>%
summarize(b_i = mean(rating - mu),
.groups='keep')

predicted_ratings_1 <- mu + test %>%
left_join(b_i, by='movieId') %>%
pull(b_i)
rmse_movie <- RMSE(test$rating, predicted_ratings_1)
model_scores <- rbind(model_scores, c("Movie effects", rmse_movie))


# Effects of User bias on rating
b_u <- train %>%
left_join(b_i, by='movieId') %>%
group_by(userId) %>%
summarize(b_u = mean(rating - mu - b_i),
.groups='keep')

predicted_ratings_2 <- test %>%
left_join(b_i, by='movieId') %>%
left_join(b_u, by='userId') %>%
mutate(pred = mu + b_i + b_u) %>%
pull(pred)
rmse_user <- RMSE(test$rating, predicted_ratings_2)
model_scores <- rbind(model_scores, c("User effects", rmse_user))


# Effects of Genres bias on rating - not required
b_g <- train %>%
left_join(b_i, by='movieId') %>%
left_join(b_u, by='userId') %>%
group_by(genres) %>%
summarize(b_g = mean(rating - mu - b_i - b_u),
.groups='keep')

predicted_ratings <- test %>%
left_join(b_i, by='movieId') %>%
left_join(b_u, by='userId') %>%
left_join(b_g, by='genres') %>%
mutate(pred = mu + b_i + b_u + b_g) %>%
pull(pred)
rmse_genres = RMSE(test$rating, predicted_ratings)
model_scores <- rbind(model_scores, c("Genres", rmse_genres))



# Effects of Year Released bias on rating - not required
b_ry <- train %>%
left_join(b_i, by='movieId') %>%
left_join(b_u, by='userId') %>%
left_join(b_g, by='genres') %>%
group_by(year_released) %>%
summarize(b_ry = mean(rating - mu - b_i - b_u - b_g),
.groups='keep')

predicted_ratings <- test %>%
left_join(b_i, by='movieId') %>%
left_join(b_u, by='userId') %>%
left_join(b_g, by='genres') %>%
left_join(b_ry, by='year_released') %>%
mutate(pred = mu + b_i + b_u + b_g + b_ry) %>%
pull(pred)
rmse_releaseyear = RMSE(test$rating, predicted_ratings)
model_scores <-rbind(model_scores, c("Release year", rmse_releaseyear))


# Effects of Age bias on rating - not required
b_a <- train %>%
left_join(b_i, by='movieId') %>%
left_join(b_u, by='userId') %>%
left_join(b_g, by='genres') %>%
left_join(b_ry, by='year_released') %>%
group_by(age_at_rating) %>%
summarize(b_a = mean(rating - mu - b_i - b_u - b_g - b_ry),
.groups='keep')

predicted_ratings_3 <- test %>%
left_join(b_i, by='movieId') %>%
left_join(b_u, by='userId') %>%
left_join(b_g, by='genres') %>%
left_join(b_ry, by='year_released') %>%
left_join(b_a, by='age_at_rating') %>%
mutate(pred = mu + b_i + b_u + b_g + b_ry + b_a) %>%
pull(pred)
rmse_age_at_rating = RMSE(test$rating, predicted_ratings)
model_scores <-rbind(model_scores, c("Age_at_rating", rmse_age_at_rating))


# --------------
# Regularization
# --------------
# Generating a sequence of values for lambda which cover minimum in small steps, but have
# large steps as we step away from minimum
lambdas <- sort(c(seq(0,     0.1, 0.1),
                  seq(0.2,   0.6, 0.025), 
                  seq(0.62,  0.8, 0.2),
                  seq(0.825, 4,   0.25)))


# lambda is applied to function that calculates RMSE for each lambda.
rmses_regularization <- sapply(lambdas, function(lambda) {
  
  mu <- mean(train$rating)
  
  # Adding movie effects
  b_i <- train %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n() + lambda),
              .groups='keep')
  
  # Adding user effects
  b_u <- train %>% 
    left_join(b_i, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n() + lambda),
              .groups='keep')
  
  # Adding genres effects
  b_g <- train %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u)/(n() + lambda), 
              .groups='keep')
  
  # Adding year_released effects
  b_ry <- train %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_g, by='genres') %>%
    group_by(year_released) %>%
    summarize(b_ry = sum(rating - mu - b_i - b_u - b_g)/(n() + lambda), 
              .groups='keep')
  
  # Adding age_at_rating effects
  b_a <- train %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_g, by='genres') %>%
    left_join(b_ry, by='year_released') %>%
    group_by(age_at_rating) %>%
    summarize(b_a = sum(rating - mu - b_i - b_u - b_g - b_ry)/(n() + lambda), 
              .groups='keep')
  
  
  # Predicted rating
  predicted_ratings <- test %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_g, by='genres') %>%
    left_join(b_ry, by='year_released') %>%
    left_join(b_a, by='age_at_rating') %>%
    mutate(pred = mu + b_i + b_u + b_g + b_ry + b_a) %>%
    pull(pred)
  
  RMSE(test$rating, predicted_ratings)
})

# Find lambda minima
lambda <- lambdas[which.min(rmses_regularization)]

# ---------------------------------
# Training on whole edx1 data set
# ---------------------------------

# Training edx data set using all the existing features developed using test and
# train.

mu <- mean(edx1$rating)


# Adding movie effects
b_i <- edx1 %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n() + lambda),
            .groups='keep')


# Adding user effects
b_u <- edx1 %>% 
  left_join(b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n() + lambda),
            .groups='keep')


# Adding genres effects
b_g <- edx1 %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u)/(n() + lambda), 
            .groups='keep')


# Adding year_released effects
b_ry <- edx1 %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_g, by='genres') %>%
  group_by(year_released) %>%
  summarize(b_ry = sum(rating - mu - b_i - b_u - b_g)/(n() + lambda), 
            .groups='keep')


# Adding age_at_rating effects
b_a <- edx1 %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_g, by='genres') %>%
  left_join(b_ry, by='year_released') %>%
  group_by(age_at_rating) %>%
  summarize(b_a = sum(rating - mu - b_i - b_u - b_g - b_ry)/(n() + lambda), 
            .groups='keep')


# Predicting the model
validation1 <- validation %>% 
  mutate(year_rated = year(as_datetime(timestamp)))%>% 
  mutate(title = str_replace(title,"^(.+)\\s\\((\\d{4})\\)$","\\1__\\2" )) %>% 
  separate(title,c("title","year_released"),"__") %>%
  select(-timestamp) %>%
  mutate(age_at_rating= as.numeric(year_rated)-as.numeric(year_released))


# -----------------
# Predicted rating
# -----------------
predicted_ratings <- validation1 %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_g, by='genres') %>%
  left_join(b_ry, by='year_released') %>%
  left_join(b_a, by='age_at_rating') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_ry + b_a) %>%
  pull(pred)


rmse_target <- 0.86490

# ----------------------
# Final Model Evaluation
# ----------------------
final_rmse <- RMSE(validation1$rating, predicted_ratings)
print(paste("Final RMSE is", final_rmse), quote = FALSE)
