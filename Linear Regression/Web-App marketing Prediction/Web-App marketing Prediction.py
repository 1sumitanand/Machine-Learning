# -*- coding: utf-8 -*-
"""
@author: Sumit Anand
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Fetch the Data
customers = pd.read_csv("Flipkart Customers")

#Entire dataframe details- rows & columns
customers.head()

#Count of rows, columns & datatype of each feature
customers.info()

#Statistical info of dataframe
customers.describe()

""" Exploratory Data Analysis """
#Check column name spacing
customers.columns

# More time on site, more money spent.
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)

sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)

#Jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership
sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)

#Data Visualization - Histogram, Corelation & scatter plots
sns.pairplot(customers)
#From data it seems Length of Membership has most correlated feature with Yearly Amount Spent

#linear model plot (using seaborn's lmplot) of Yearly Amount Spent vs. Length of Membership. 
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)

""" Training and Testing Data """
y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]

#Using model_selection.train_test_split from sklearn to split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


""" Training and Testing Data """
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

# The coefficients
print('Coefficients: \n', lm.coef_)

""" Predicting Test Data """

predictions = lm.predict( X_test)

#scatterplot of the real test values versus the predicted values.
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

""" Evaluating the Model """

# Accuracy of model
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('RSquare:', metrics.r2_score(y_test, predictions))

#histogram of the residuals and make sure it looks normally distributed
sns.distplot((y_test-predictions),bins=50);

#interpreting the coefficients to get an idea from the data
coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients