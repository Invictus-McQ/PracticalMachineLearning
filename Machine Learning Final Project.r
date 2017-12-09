setwd("~")
dev.off()
rm(list = ls())

##options(warn=-1)
if (!require("caret")) {install.packages("caret")}
library(caret)
if (!require("randomForest")) {install.packages("randomForest")}
library(randomForest)
if (!require("e1071")) {install.packages("e1071")}
library(e1071)
if (!require("plyr")) {install.packages("plyr")}
library(plyr)
if (!require("reshape2")) {install.packages("reshape2")}
library(reshape2)
if (!require("plotly")) {install.packages("plotly")}
library(plotly)



mainDir <- "~."
subDir <- "R"
subDir2 <- "Johns Hopkins Data Science Certification"
subDir3 <- "Practical Machine Learning"
if (file.exists(subDir)) { setwd(file.path(mainDir, subDir))} 
    else {  dir.create(file.path(mainDir, subDir), showWarnings = FALSE) 
      setwd(file.path(mainDir, subDir))}
  mainDir <- getwd()
if (file.exists(subDir2)) { setwd(file.path(mainDir, subDir2))} 
    else {dir.create(file.path(mainDir, subDir2), showWarnings = FALSE) 
      setwd(file.path(mainDir, subDir2))}
  mainDir <- getwd()
if (file.exists(subDir3)) {setwd(file.path(mainDir, subDir3))} 
    else {dir.create(file.path(mainDir,subDir3), showWarnings = FALSE) 
      setwd(file.path(mainDir, subDir3))}
  
trainFileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
trainFileName <- "pml-training.csv"
  
if (!file.exists(trainFileName)) {
    download.file(trainFileURL ,trainFileName,method="auto") }
train <- read.csv(trainFileName, header = TRUE, sep = ",", quote = "\"")
  
testFileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
testFileName <- "pml-testing.csv"
  
if (!file.exists(testFileName)) {
    download.file(testFileURL ,testFileName,method="auto") }
test <- read.csv(testFileName, header = TRUE, sep = ",", quote = "\"")

train <- subset(train, select=-c(1:7))
test <- subset(test, select=-c(1:7))

str(train)
# Putting a Threshold value for the check on the number of occurences of 'NA' or '""' empty values in the datasets
train_threshold_val <- 0.95 * dim(train)[1]

#Remove columns which have more than 95 % NA or empty values
include_cols <- !apply(train, 2, function(x) sum(is.na(x)) > train_threshold_val || sum(x=="") > train_threshold_val)
train <- train[, include_cols]

#Remove columns with very low variance values
nearZvar <- nearZeroVar(train, saveMetrics = TRUE)
train <- train[ , nearZvar$nzv==FALSE] 
dim(train)

# Remove columns highly correlated with one another via correlation matrix 
Corr_Matrix <- abs(cor(train[,-dim(train)[2]]))
diag(Corr_Matrix) <- 0
HighlyCorrelated_col <- findCorrelation(Corr_Matrix, verbose = FALSE , cutoff = .95)
train <- train[, -c(HighlyCorrelated_col)]
dim(train)

set.seed(1338)
#Partition the training dataset into 2 parts with p=0.66
Partition = createDataPartition(train$classe, p = 0.66, list=FALSE)
train1 <- train[Partition,]
train2 <- train[-Partition,]


# Random Forest Model.  Using 'importance=TRUE' for use of the Variable-Importance plot.
RFModel <- randomForest(classe ~., data=train1, importance=TRUE, ntree=250)
RFModel

#Frequency Table of the predicted values for the test set
train2_pred <- predict(RFModel, newdata=train2)
train2_table <- count(train2_pred)
train2_table$Percent <- train2_table$freq / sum(train2_table$freq)
train2_table

# Showing the Confusion Matrix here :
ConfusMatrix <- confusionMatrix(train2_pred, train2$classe)
ConfusMatrix

##Classification Accuracy:
sum(diag(ConfusMatrix$table))/sum(ConfusMatrix$table)

RFImportance <- varImp(RFModel, scale=FALSE)
RFImportance$Category <- row.names(RFImportance)
RFImportanceMelt <- melt(RFImportance, "Category")
RFImportance <- RFImportance[order(RFImportance$Category),]
RFImportanceMelt <- RFImportanceMelt[order(RFImportanceMelt$Category),]


varImpPlot(RFModel,  main="Top 25 Variable Importance")


f1 <- list(
  family = "Arial, sans-serif",
  size = 14,
  color = "black"
)
f2 <- list(
  family = "Old Standard TT, serif",
  size = 9,
  color = "black"
)
a <- list(
  title = "Names of the Columns of the Training Dataset after Tidying",
  titlefont = f1,
  showticklabels = TRUE,
  tickangle = 90,
  tickfont = f2
)

b <- list(
  title = "Value of Importance determined by the Random Forest Algorithm",
  titlefont = f1,
  showticklabels = TRUE,
  tickfont = f2
)

m = list(
  l = 50,
  r = 40,
  b = 110,
  t = 40,
  pad = 0
) 

plot_ly(data = RFImportanceMelt, x = ~Category, y = ~value, 
        type = 'scatter', mode = 'lines', color = ~variable)  %>%
  layout(xaxis = a, yaxis = b, 
    autosize = T, margin = m, title = "Values of Importance for Each Column \n Imagined as a Line")

##PartialPlots
 imp <- importance(RFModel)
 impvar <- rownames(imp)[order(imp[, 1], decreasing=TRUE)]
 dev.new(width=7, height=7)
 par(mar=c(1.5, 1.5, 1.2, 1.2))
 op <- par(mfrow=c(7, 7))
 for (i in seq_along(impvar)) {
   partialPlot(RFModel, train1, impvar[i], xlab=impvar[i],
               main=paste("Partial Dependence on", impvar[i]),
)
 }
 par(op)

##Verifying the results with the test dataset:
test_pred <- predict(RFModel, newdata=test)
test_table <- count(test_pred)
test_table$Percent <- test_table$freq / sum(test_table$freq)
test_table
