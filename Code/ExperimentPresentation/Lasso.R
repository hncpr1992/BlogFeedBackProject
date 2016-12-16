setwd("/home/zhoutao/Desktop/STAT542/project/Data")
library(glmnet)
train.file <- "blogData_train.csv"
test.file <- "blogData_test.csv"
train <- read.csv(train.file, header = F)
colnames(train) <- c(paste('V',seq(1,280), sep=''), 'y')
neg_log <- function(x) {
  if(x >= 0) {
    return (log(x+1))
  } else {
    return (-log(-x+1))
  }
}
train.x <- subset(train, select = -y)
log_col <- paste('V', c(1:60, 278:280), sep='')
log_col_neg <- paste('V',c(21,22,23,24,25,46,47,48,49,50,55,60), sep='')
for (col in log_col) {
  if (col %in% log_col_neg) {
    train.x[, col] = sapply(train.x[, col], neg_log)
  } else {
    train.x[, col] = log(train.x[, col] + 1)
  }
}
train.x <- data.matrix(train.x)

train.y <- train$y
train.y <- log(train.y + 1)
train.y <- data.matrix(train.y)

set.seed(542)
fit <- cv.glmnet(train.x, train.y, nfolds=10, family='gaussian', alpha=1) 
plot(fit)
### get selected feature
lambda <- fit$lambda
n_best <- which(lambda == fit$lambda.1se)
beta <- fit$glmnet.fit$beta
n <- dim(beta)[2]
beta_best <- beta[, n_best]
nonZeroVar <- names(beta_best)[beta_best != 0]
origVar <- paste('V', seq(1,280), sep='')
origVar <- origVar[origVar %in% colnames(train.x)]
# useful added feature
nonZeroAddedVar <- nonZeroVar[!(nonZeroVar %in% origVar)]
# unusful original feature
zeroOrigVar <- origVar[!(origVar %in% nonZeroVar)]
# write selected variables to file
for (pat in nonZeroAddedVar) {
  cat(paste(pat, '\n', sep=''), file="UsefulAddedVar.txt", append = T)
}
for (pat in zeroOrigVar) {
  cat(paste(pat, '\n', sep=''), file="UnusefulOrigVar.txt", append = T)
}

# check regression accuracy
test <- read.csv(test.file, header = F)
colnames(test) <- c(paste('V',seq(1,280), sep=''), 'y')

test.x <- subset(test, select = -y)
for (col in log_col) {
  if (col %in% log_col_neg) {
    test.x[, col] = sapply(test.x[, col], neg_log)
  } else {
    test.x[, col] = log(test.x[, col] + 1)
  }
}
test.x <- data.matrix(test.x)

test.y <- test$y
test.y <- log(test.y+1)
test.y <- data.matrix(test.y)

pred <- predict(fit ,newx=test.x, s="lambda.1se")
error <- sum((test.y - pred)^2)/length(test.y)
error