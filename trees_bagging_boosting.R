#####
# Machine Learning and Predictive Analytics
# Fall 2016
# Lecture 5: Trees, Bagging, Forests, and Boosting
# Angelo Mancini
#
# Description: This script explores trees, bagging, and boosting using a data set
# that from crash tests.  The response is the acceleration of the helment during the crash
# and the only predictor is the amount of time since impact.  We do not explore random forests
# in this script because when there is only one predictor, it's equivalent to bagging.  They're
# also covered in the lab in the text (R), and we'll cover them in Python in the next hands-on
# session.
#####


#####
# Load libraries
#####

# Note: You'll need to have the 'tree' and 'kknn'
#       libraries installed.

# 'tree' library for decision trees
library(tree)

# 'kknn' library for k-nearest neighbors regression
library(kknn)

###
# Part A: Linear Model vs. Trees vs. K-NN
###

# Import the data and take a look
cycle_data <- read.csv('mcycle.csv',header=TRUE)
head(cycle_data)

# Let's remove the 'X' column, which is redundant
cycle_data$X <- NULL

# Next, let's plot the cycle data
plot(cycle_data$times,cycle_data$accel,
     main="Acceleration vs. Time Since Impact",
     xlab="Time Since Impact",
     ylab="Acceleration of Helmet")

# We see that this data is highly non-linear, which will pose a problem
# for linear models.  Let's confirm.
cycle_glm <- glm(accel ~ times, data=cycle_data, family='gaussian')
lines(cycle_data$times,cycle_glm$fitted.values,type='l',col='red')

# Let's see if adding a quadratic term helps
cycle_glm_quad <- glm(accel~ times + I(times^2), data=cycle_data, family='gaussian')
lines(cycle_data$times,cycle_glm_quad$fitted.values,type='l',col='blue')

# Quadratic term doesn't really help.  Let's move on to a tree.

# Build the tree, plot it, and label it.
cycle_tree <- tree(accel ~ times, data=cycle_data)
plot(cycle_tree)
text(cycle_tree)

# Next, let's plot the tree's predictions to see if it picks up on the
# nonlinear structure in the data
cycle_tree_preds <- predict(cycle_tree, newdata = data.frame(times=cycle_data$times))
plot(cycle_data$times,cycle_data$accel,
     main="Acceleration vs. Time Since Impact",
     xlab="Time Since Impact",
     ylab="Acceleration of Helmet")
points(cycle_data$times,cycle_tree_preds,col='blue')
lines(cycle_data$times,cycle_tree_preds,type='l',col='blue')

# We can also use cross-validation to prune the tree
cycle_tree_cv <- cv.tree(cycle_tree,K=3)
plot(cycle_tree_cv)
# Values along the top refer to the penalization term in
# cost complexity pruning (~ alpha in the ISLR)

# We see that for this tree, the deepest tree (all 8 leaf nodes)
# does best.  Makes sense since this is highly nonlinear data.

# Let's compare this to k-nearest neighbors, the only other non-parametric
# approach for regression we've seen so far.
cycle_knn <- kknn(accel ~ times, train=cycle_data, test=cycle_data, k = 4, kernel = 'rectangular')
points(cycle_data$times, cycle_knn$fitted.values,type='o',col='red')

# We see that k-nearest neighbors and a regression tree both do much betteer
# than a linear or even quadratic model in-sample.  This makes sense since the
# data isn't very linear, or even smooth enough for a good polynomial
# regression.

###
# Part B: Bagging
#
# Description: Bagging is a general approach that can be used with any kind of model,
# not just trees.  Even though it tends to work best with trees, we'll 
# try it with some other models to get practice and build intuition.
###

## Let's try it first with a linear model.  This means that, for every bootstrap
## sample, we'll build a linear model.  We'll then take our final predictions
## to be the average prediction across all the bootstrapped linear models.

# Let's set the number of trees we want to use
B <- 300

# Let's store the number of observations, to make coding easier
n <- nrow(cycle_data)

# Let's create a data frame to store the predictions at each
# training observation for each bootstrapped model
bagged_preds_linear <-  data.frame(matrix(0,nrow(cycle_data),B))

# Now, let's go into the bagging loop
for (b in 1:B) {
  
  # Draw a bootstrap sample -sampling with replacement
  b_sample <- sample(1:n,n,replace=TRUE)
  
  # Fit a linear model on the bootstrap sample
  b_glm <- glm(accel ~ times, data=cycle_data[b_sample,], family='gaussian')
  
  # Predict on all the training observations using linear model 
  # we just fit or this bootstrap sample
  b_pred <- predict(b_glm,newdata=data.frame(times=cycle_data$times))
  
  # Store the predictions
  bagged_preds_linear[,b] <- b_pred
  
}

# Now, let's plot each of the bootstrapped models, one by one
plot(cycle_data$times,cycle_data$accel,
     main="Acceleration vs. Time Since Impact",
     xlab="Time Since Impact",
     ylab="Acceleration of Helmet")

for (b in 1:B) {
  
  lines(cycle_data$times,bagged_preds_linear[,b],col='red')
  
  cat ("Press [enter] to continue")
  line <- readline()
  
  if(line == 'b') {break}
}

# Next, let's plot the final bootstrapped model at each of the
# training points.
b_glm_final_preds <- rowSums(bagged_preds_linear)/B
lines(cycle_data$times,b_glm_final_preds,col='blue')

# We see that bagging with a linear model isn't that helpful.
# Reducing the variance of a highly biased model only buys you so much.

## Now, let's try bagging with k-nearest neighbors.
## We'll try using a very flexible model, e.g., k=1 or 2, since
## bagging works by reducing variance of low-bias/high-variance models
## through averaging.

# Let's first set the number of neighbors we want to use
k <- 3

# The bagging algorithm is largely unchanged. We only need to change
# the glm fit to a nearest neighbors fit, as well as how we make predictions.
# For clarity, I'll reinitialize the bagging parameters.

# Let's set the number of trees we want to use
B <- 300

# Let's create a data frame to store the predictions at each
# training observation for each bootstrapped model
bagged_preds_knn <-  data.frame(matrix(0,nrow(cycle_data),B))

# Now, let's go into the bagging loop
for (b in 1:B) {
  
  # Draw a bootstrap sample -sampling with replacement
  b_sample <- sample(1:n,n,replace=TRUE)
  
  # Fit and predict with a knn-model. We'll use three neighbors.
  b_knn <- kknn(accel ~ times, train=cycle_data[b_sample,], test=cycle_data, k = k, kernel = 'rectangular')
  bagged_preds_knn[,b] <- b_knn$fitted.values
  
}

# Now, let's plot each of the bootstrapped models, one by one
plot(cycle_data$times,cycle_data$accel,
     main="Acceleration vs. Time Since Impact",
     xlab="Time Since Impact",
     ylab="Acceleration of Helmet")

for (b in 1:B) {
  
  lines(cycle_data$times,bagged_preds_knn[,b],col='red')
  
  cat ("Press [enter] to continue")
  line <- readline()
  
  if(line == 'b') {break}
}

# Next, let's plot the final bootstrapped model at each of the
# training points.
b_knn_final_preds <- rowSums(bagged_preds_knn)/B
lines(cycle_data$times,b_knn_final_preds,col='blue',lwd=5)

## We see that bagging with k-nearest neighbors using a flexible approach
## (k=3) works much better than with a linear model.  What happens if we use a less
## less flexible approach, e.g., k=7?

for (b in 1:B) {
  
  # Draw a bootstrap sample -sampling with replacement
  b_sample <- sample(1:n,n,replace=TRUE)
  
  # Fit and predict with a knn-model. We'll use three neighbors.
  b_knn <- kknn(accel ~ times, train=cycle_data[b_sample,], test=cycle_data, k = 7, kernel = 'rectangular')
  bagged_preds_knn[,b] <- b_knn$fitted.values
  
}

b_knn_final_preds <- rowSums(bagged_preds_knn)/B
lines(cycle_data$times,b_knn_final_preds,col='red',lwd=5)

## As expected, we get a smoother model fit. 

## Now, let's do bagging with trees. Again, very little will change -we'll just need to
## adjust how we fit a model on each bootstrapped sample, and how we predict.

# Let's set the number of trees we want to use
B <- 300
n <- nrow(cycle_data)
# Let's create a data frame to store the predictions at each
# training observation for each bootstrapped model
bagged_preds_trees <-  data.frame(matrix(0,nrow(cycle_data),B))

# Now, let's go into the bagging loop
for (b in 1:B) {
  
  # Draw a bootstrap sample -sampling with replacement
  b_sample <- sample(1:n,n,replace=TRUE)
  
  # Fit on the bootstrap sample
  b_tree <- tree(accel ~ times, data=cycle_data[b_sample,])
  
  # Predict on the training obsevations
  b_tree_preds <- predict(b_tree, newdata = data.frame(times=cycle_data$times))
  
  # Store the predictions
  bagged_preds_trees[,b] <- b_tree_preds
  
}

# Now, let's plot each of the bootstrapped models, one by one
plot(cycle_data$times,cycle_data$accel,
     main="Acceleration vs. Time Since Impact",
     xlab="Time Since Impact",
     ylab="Acceleration of Helmet")

for (b in 1:B) {
  
  lines(cycle_data$times,bagged_preds_trees[,b],col='red')
  
  cat ("Press [enter] to continue")
  line <- readline()
  
  if(line == 'b') {break}
}

# Next, let's plot the final bootstrapped model at each of the
# training points.
b_tree_final_preds <- rowSums(bagged_preds_trees)/B
lines(cycle_data$times,b_tree_final_preds,col='blue',lwd=5)

## Looks pretty good.  Compared to the bagged k-nearest neighbors, seems
## likes it's overfitting less in the low and high regions (time <= 15 and time >= 40).

###
# Part C: Boosting
#
# Description: Next, we'll look at boosting.  Instead of building many low bias models and 
# averaging out the overfitting, boosting instead builds a sequence of simple models in which
# each model tries to make a small improvement on the previous models
###

## Like bagging, boosting can be used with any model, but it's most common to use trees. We'll
## code it up manually here to build intuition. In practice, we'd use the 'gbm' package in R.

# First, let's set the parameters used by boosting
B <- 600 # Number of trees
lambda <- 0.01 # Shrinkage term
d <- 1  # Interaction depth

# Second, let's initialize the set of residuals, which get upated
# after each step in boosting.  They should initially be set to the
# response values we're trying to fit
r <- cycle_data$accel

# Third, let's crete an object to store the boosted fits at each step
# We add one additional column for the 'null' model (all predictions= 0)
boost_fits <- data.frame(matrix(0,nrow(cycle_data),B+1))

## Now, let's go into the boosting loop
for (b in 2:(B+1)) {
  
  # Fit a tree to the current residuals
  b_tree <- tree(r ~ times, data=data.frame(r=r,times=cycle_data$times))
  
  # Prune the tree back to just (d+1) noes -shouldn't have to do this but 'tree' package
  # doesn't let us specify interaction depth.
  
  if (sum(class(b_tree)=='singlenode') == 0) {
  
  b_tree <- prune.tree(b_tree,best=(d+1))
  
  }
  
  # Let's predict on the training set using the newest tree
  b_tree_preds <- predict(b_tree, newdata = data.frame(times=cycle_data$times))
  
  # Store the predictions
  boost_fits[,b] <- b_tree_preds
  
  # Update the residuals
  r <- r - lambda*b_tree_preds
  
}

# Now, let's plot the predictions of the model at each step in boosting.
plot(cycle_data$times,cycle_data$accel,
     main="Acceleration vs. Time Since Impact",
     xlab="Time Since Impact",
     ylab="Acceleration of Helmet")

for (b in 2:(B+1)) {
  
  lines(cycle_data$times,lambda*rowSums(boost_fits[,1:b]),col='red')
  
  cat ("Press [enter] to continue")
  line <- readline()
  
  if(line == 'b') {break}
}

# Next, let's plot the final bootstrapped model at each of the
# training points.
b_boost_final_preds <- lambda*rowSums(boost_fits)
lines(cycle_data$times,b_boost_final_preds,col='blue',lwd=5)

