setwd("/home/zhoutao/Desktop/STAT542/project/BlogFeedback/")
train <- read.csv("blogData_train.csv",header=F)
require(ggplot2)
require(corrplot)
##Zero percentage
isZero <- sum(train$V281 == 0)
isNonZero <- sum(train$V281 != 0)
percent <- paste(as.character(round(c(isZero, isNonZero)/nrow(train)*100,2)),"%",sep='')
pie(c(isZero,isNonZero), labels=percent, col=c("skyblue","darkorange"))
legend("top", legend=c("Zero","NonZero"), fill=c("skyblue","darkorange"),horiz=T,bty='n')
## NonZero distribution
foo <- train$V281
foo <- foo[foo != 0]
ggplot(data.frame(x=foo), aes(x)) + 
  geom_histogram(bins=200) +
  labs(x="Nonzero comments in next 24h", y="Counts")+
  theme(text = element_text(size=14))+
  scale_y_sqrt()
sum(foo<10)/length(foo) # 0.7470007
sum(foo<20)/length(foo) # 0.8447287
## correlation
corrs <- c()
for (i in 1:280) {
  corrs <- c(corrs, cor(train[,i], train[, 281]))
}
corrs[is.na(corrs)] <- 0
corrs <- abs(corrs)
order(corrs, decreasing = T)[1:10] # 10 21  6  5 11 15 20  1 52 16
# median of the number of comments in the last 24h
# average of difference of number of comments in last 24h and 24-48h
# average of number of comments in the last 24h
# median of total number of comments before basetime
# average of number of comments between last 24-48h

corrs[order(corrs,decreasing = T)[1:10]]
#  [1] 0.5065403 0.5033746 0.4976313 0.4917072 0.4901115 0.4896736 0.4863156
#  [8] 0.4854641 0.4720608 0.4719988

# make correlation matrix plot
foo <- train[,c(10,21,6,5,11,281)]
corrplot.mixed(cor(foo), upper="ellipse")

# compare comment and link
c(cor(train$V51,train$V281), cor(train$V56,train$V281)) # 0.3144457 0.1919169
c(cor(train$V52,train$V281), cor(train$V57,train$V281)) # 0.4720608 0.2609028
c(cor(train$V53,train$V281), cor(train$V58,train$V281)) # 0.11764232 0.06714093
c(cor(train$V54,train$V281), cor(train$V59,train$V281)) # 0.3141766 0.1986375
c(cor(train$V55,train$V281), cor(train$V60,train$V281)) # 0.2962728 0.1461449

# length of time published and length of post
cor(train$V61, train$V281) # -0.1529081
cor(train$V62, train$V281) # 0.04820917

# words
foo <- train[, 63:262]
word_freq <- colSums(foo)
plot(word_freq, type="h", col="blue4", main="Word Frequency", ylab="Count", xlab="", cex.lab=1.2)
points(word_freq, pch=16, col="blue4",cex=0.6)
sum(word_freq<523) # 117, more than half the words only appear in less than 1% of the blog post
# conditional probability
foo <- c()
for (i in 63:262) {
  supAB <- sum(train[, i] == 1 & train[, 281] > 0)
  supA <- sum(train[, i] == 1)
  foo <- c(foo, supAB/supA)
}
plot(foo, type="h", col="blue4", main="Conditional probability", ylab="Conditional Probability", xlab="", cex.lab=1.2)
points(foo, pch=16, col="blue4",cex=0.6)

# influence of weekday
basetimeday <- rep(0,nrow(train))
publishday <- rep(0,nrow(train))
for ( i in 1:7) {
  basetimeday[train[, (262+i)] == 1] <- i
  publishday[train[, (269+i)] == 1] <- i
}
foo <- data.frame(basetimeday=basetimeday, publishday=publishday, comment=log10(train[, 281]+1))
ggplot(foo, aes(factor(basetimeday), comment))+
  geom_boxplot()+
  labs(x="Weekday of the Basetime", y="log Count")+
  theme(text=element_text(size=14))

ggplot(foo, aes(factor(publishday), comment))+
  geom_boxplot()+
  labs(x="Weekday of the Publication day", y="log Count")+
  theme(text=element_text(size=14))

# rule mining
# rule1 
foo <- sum(train$V51 == 0 & train$V61 > 12)
bar <- sum(train$V51 == 0 & train$V61 > 12 & train$V281 == 0)
foo/nrow(train) # 0.2335248
bar/foo # 0.9521085
# rule2
foo <- sum(train$V52 == 0 & train$V51 != 0)
bar <- sum(train$V52 == 0 & train$V51 != 0 & train$V281 == 0)
foo/nrow(train) # 0.1892475
bar/foo # 0.8524607
