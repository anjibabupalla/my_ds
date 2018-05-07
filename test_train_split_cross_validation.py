# Databricks notebook source
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
columns = "age sex bmi map tc ldl hdl tch ltg glu".split()
#print(columns)
diabetes = datasets.load_diabetes()
#print(diabetes.data)
df = pd.DataFrame(diabetes.data,columns=columns)
#print(df.head())
y1 = diabetes.target # define the target variable (dependent variable) as y
#print(y1)

# COMMAND ----------

#The test_size=0.2 inside the function indicates the percentage of the data that should be held over for testing. It’s usually around 80/20 or 70/30.
X_train,X_test,Y_train,Y_test = train_test_split(df,y,test_size=0.2)
#print(X_train.head())
#print(Y_train)
#print(X_train.shape,Y_train.shape)
#print(X_test.shape,Y_test.shape)


# COMMAND ----------

lm = linear_model.LinearRegression()
#we’re fitting the model on the training data and trying to predict the test data
model = lm.fit(X_train,Y_train)
predictions= lm.predict(X_test)
predictions[0:5]

# COMMAND ----------

plt.scatter(Y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")

# COMMAND ----------

print ("Score:",model.score(X_test, Y_test))

# COMMAND ----------

#But train/test split does have its dangers — what if the split we make isn’t random? What if one subset of our data has only people from a certain state, employees with a certain income level but not other income levels, only women or only people at a certain age? (imagine a file ordered by one of these). This will result in overfitting, even though we’re trying to avoid it! This is where cross validation comes in.
#cross validation: It’s very similar to train/test split, but it’s applied to more subsets. Meaning, we split our data into k subsets, and train on k-1 one of those subset. What we do is to hold the last subset for test
#There are a bunch of cross validation methods, I’ll go over two of them: the first is K-Folds Cross Validation and the second is Leave One Out Cross Validation (LOOCV).The disadvantage of this method is that the training algorithm has to be rerun from scratch k times, which means it takes k times as much computation to make an evaluation.A variant of this method is to randomly divide the data into a test and training set k different times. The advantage of doing this is that you can independently choose how large each test set is and how many trials you average over.
#In K-Folds Cross Validation we split our data into k different subsets (or folds). We use k-1 subsets to train our data and leave the last subset (or the last fold) as test data. We then average the model against each of the folds and then finalize our model. After that we test it against the test set
#LOOCV: Leave out one cross validation: In this type of cross validation, the number of folds (subsets) equals to the number of observations we have in the dataset. We then average ALL of these folds and build our model with the average. We then test the model against the last fold. Because we would get a big number of training sets (equals to the number of samples), this method is very computationally expensive and should be used on small datasets.

# COMMAND ----------

from sklearn.model_selection import KFold
import numpy as np
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]]) # create an array
Y = np.array([1, 2, 3, 4]) # Create another array
kf = KFold(n_splits=2) # Define the split - into 2 folds
kf.get_n_splits(X)# returns the number of splitting iterations in the cross-validator
#print(kf)
for train_index, test_index in kf.split(X):
 print("TRAIN:", train_index, "TEST:", test_index)


# COMMAND ----------

from sklearn.model_selection import LeaveOneOut
X = np.array([[1, 2], [3, 4]])
Y = np.array([1, 2])
loo = LeaveOneOut()
loo.get_n_splits(X)
print(X)
for train_index, test_index in loo.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)

# COMMAND ----------

#se the cross_val_predict function to return the predicted values for each data point when it’s in the testing slice.
from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn import metrics
# Perform 6-fold cross validation
scores = cross_val_score(model,df,y1,cv=6)
print("Cross-validated scores:", scores)
#As you can see, the last fold improved the score of the original model — from 0.485 to 0.569
predictions = cross_val_predict(model, df, y, cv=6)
plt.scatter(y1, predictions)

# COMMAND ----------

#let’s check the R² score of the model (R² is a “number that indicates the proportion of the variance in the dependent variable that is predictable from the independent variable(s)”. Basically, how accurate is our model)
accuracy = metrics.r2_score(y1, predictions)
print("Cross-Predicted Accuracy:", accuracy)
