setwd("/home/zhoutao/Desktop")
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
log_col <- paste('V', c(1:60, 62, 278:280), sep='')
log_col_neg <- paste('V',c(21,22,23,24,25,46,47,48,49,50,55,60), sep='')
for (col in log_col) {
  if (col %in% log_col_neg) {
    train.x[, col] = sapply(train.x[, col], neg_log)
  } else {
    train.x[, col] = log(train.x[, col] + 1)
  }
}
# reorder column to keep them in group order
group <- 1:280
group[seq(1,21,5)] = 1
group[seq(2,22,5)] = 2
group[seq(3,23,5)] = 3
group[seq(4,24,5)] = 4
group[seq(5,25,5)] = 5
group[seq(26,46,5)] = 6
group[seq(27,47,5)] = 7
group[seq(28,48,5)] = 8
group[seq(29,49,5)] = 9
group[seq(30,50,5)] = 10
group[51:55] = 11
group[56:60] = 12
group[61:62] = 13
group[63:262] = 14
group[263:269] = 15
group[270:276] = 16
group[277:280] = 17
train.x.gporder <- train.x[, which(group==1)]
for (i in 2:17) {
  train.x.gporder <- cbind(train.x.gporder, train.x[, which(group==i)])
}
group = sort(group)
X <- data.matrix(train.x.gporder)

train.y <- train$y
train.y <- log(train.y + 1)
y <- data.matrix(train.y)

# use group lasso to select feature
# group lasso
library(gglasso) 
X <- X[, which(group != 14)]
group <- group[group != 14]
group[group == 15] <- 14
group[group == 16] <- 15
group[group == 17] <- 16
samp <- sample(52397, 10000)
X <- X[samp, ]
y <- y[samp]
lambdas <- c(1,0.1,5e-2,1e-2,8e-3,5e-3,3e-3,1e-3,5e-4,1e-4)
lambdas2 <- c(2.5e-3,2e-3,1.5e-3,8e-4,7e-4)
fit <- gglasso(X, y, group=group, loss='ls', lambda=lambdas)
fit2 <- gglasso(X, y, group=group, loss='ls', lambda=lambdas2)


