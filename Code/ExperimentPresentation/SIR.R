setwd("/home/zhoutao/Desktop/STAT542/project/Data")
library(dr)
data <- read.csv("sirdata.csv")
fit.sir = dr(V281~., data = data, method = "sir", nslices = 10)
fit.sir$evalues # use the first 4 directions
direction = fit.sir$raw.evectors[, 1:4]
colnames(direction) <- paste("D", 1:4, sep='')
write.csv(direction, "sir_direction.csv", row.names = F)
