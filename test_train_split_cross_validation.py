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
y = diabetes.target # define the target variable (dependent variable) as y
#print(y)

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
