
###########LINEAR REGRESSION
# Random data in which y is a noisy function of x
X <- runif(100, -5, 5)
y <- sin(X) + rnorm(100) + 3
# Fit a model (regress weight on height)
fit <- lm(y ~ X)
print(fit)

# beta-hat
fit_params <- fit$coefficients

# Plot
plot(x=X, y=y, cex = 1, col = "grey")

# Draw the regression line (intercept, slope)
abline(a=fit_params[[1]], b=fit_params[[2]], col="blue")

# Matrix of predictors (we only have one in this example)
X_mat <- as.matrix(X)
# Add column of 1s for intercept coefficient
intcpt <- rep(1, length(y))

# Combine predictors with intercept
X_mat <- cbind(intcpt, X_mat)

# OLS (closed-form solution)
beta_hat <- solve(t(X_mat) %*% X_mat) %*% t(X_mat) %*% y
print(beta_hat)


# Plot
plot(x=X, y=y, cex = 1, col = "grey")

# Draw the previous regression line
abline(a=fit_params[[1]], b=fit_params[[2]], col="blue")
# Current regression line
abline(a=beta_hat[[1]], b=beta_hat[[2]], col="green")

# To get y-hat:
y_hat <- X_mat %*% beta_hat
points(x=X, y=y_hat, pch = 2, col='yellow')

gradient_descent <- function(X, y, lr, epochs)
{
  X_mat <- cbind(1, X)
  # Initialise beta_hat matrix
  beta_hat <- matrix(0.1, nrow=ncol(X_mat))
  for (j in 1:epochs)
  {
    residual <- (X_mat %*% beta_hat) - y
    delta <- (t(X_mat) %*% residual) * (1/nrow(X_mat))
    beta_hat <- beta_hat - (lr*delta)
    # Draw the regression line each epoch
    abline(a=beta_hat[[1]], b=beta_hat[[2]], col="grey")
  }
  # Return 
  beta_hat
}

# Plot
plot(x=X, y=y, cex = 1, col = "grey")

beta_hat <- gradient_descent(X, y, 0.1, 200)
print(beta_hat)

# Draw the regression line
abline(a=beta_hat[[1]], b=beta_hat[[2]], col="red")

# To get y-hat:
y_hat <- X_mat %*% beta_hat
points(x=X, y=y_hat, pch = 2, col='yellow')


#############LOGISTIC REGRESSION
# Two possible outcomes -> binomial
data_df <- as.data.frame(iris)
idx <- data_df$Species %in% c("virginica", "versicolor")
data_df <- data_df[idx,]
y <- ifelse(data_df$Species=="virginica", 1, 0)

# For faster convergence let's rescale X
# So that we can plot this consider only 2 variables
X <- data_df[c(1,3)]
X <- as.matrix(X/max(X))

# Resulting data-set
head(X)
head(y)

# Fit model
model <- glm(y ~ X, family=binomial(link='logit'))

# Params
print(coef(model))
# Coefficients:
# (Intercept) XSepal.Length XPetal.Length 
# -39.83851     -31.73243     105.16992 
#summary(model)

# Visualise the decision boundary
intcp <- coef(model)[1]/-(coef(model)[3])
slope <- coef(model)[2]/-(coef(model)[3])

# Our points
plot(x=X[,1], y=X[,2], cex = 1, col=data_df$Species,
     main = "Iris type by length and width", 
     xlab = "Sepal Length", ylab = "Petal Length")

# Decision boundary
abline(intcp , slope, col='blue')

logLik(model)  # Log-likelihood

# Calculate log-likelihood ourself
log_likelihood <- function(X_mat, y, beta_hat)
{
  scores <- X_mat %*% beta_hat
  # Need to broadcast (y %*% scores)
  ll <- (y * scores) - log(1+exp(scores))
  sum(ll)
}

log_likelihood(cbind(1, X), y, coef(model))

# Calculate activation function (sigmoid for logit)
sigmoid <- function(z){1.0/(1.0+exp(-z))}

logistic_reg <- function(X, y, epochs, lr)
{
  X_mat <- cbind(1, X)
  beta_hat <- matrix(1, nrow=ncol(X_mat))
  for (j in 1:epochs)
  {
    residual <- sigmoid(X_mat %*% beta_hat) - y
    # Update weights with gradient descent
    delta <- t(X_mat) %*% as.matrix(residual, ncol=nrow(X_mat)) *  (1/nrow(X_mat))
    beta_hat <- beta_hat - (lr*delta)
  }
  # Print log-likliehood
  print(log_likelihood(X_mat, y, beta_hat))
  # Return
  beta_hat
}
curve(sigmoid, -10, 10)

# Takes a while to converge with GD!
beta_hat <- logistic_reg(X, y, 300000, 5)
print(beta_hat)

# Intercept    -39.83848
# Sepal.Length -31.73240
# Petal.Length 105.16983

# Visualise the decision boundary
plot(x=X[,1], y=X[,2], cex = 1, col=data_df$Species,
     main = "Iris type by length and width", 
     xlab = "Sepal Length", ylab = "Petal Length")

# Visualise the decision boundary
intcp <- beta_hat[1]/-(beta_hat[3])
slope <- beta_hat[2]/-(beta_hat[3])

abline(intcp , slope, col='purple')

