#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Check out the Data
HousingData = pd.read_csv('Housing_Data.csv')

#Entire dataframe details- rows & columns
HousingData.head()

#Count of rows, columns & datatype of each feature
HousingData.info()

#Statistical info of dataframe
HousingData.describe()

#Check column name spacing
HousingData.columns

#Data Visualization - Histogram, Corelation & scatter plots
sns.pairplot(HousingData)

#Distribution of price column
sns.distplot(HousingData['Price'])

#Corelation of the columns
HousingData.corr()

#Heatmap of Corelation of the columns with values
sns.heatmap(HousingData.corr(), annot=True)

##Training a Linear Regression Model

#Check column and remove Address as string const not needed
HousingData.columns

#Assign Independent variable
x = HousingData[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
#Assign dependent variable
y = HousingData['Price']

#Split data using sklearn lib 
#Train Test Split
from sklearn.model_selection import train_test_split
#Shift+Tab to read documentation in Jupyter
#train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)

#Creating and Training the Model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

#Model Evaluation
# print the intercept
print(lm.intercept_)

#Coefficient of model with all features (for multiplles values of x, let's say features)
#lm.coef
coeff_df = pd.DataFrame(lm.coef_,x.columns,columns=['Coefficient'])
coeff_df

#Predictions from our Model (based on value of x find y by putting in eq.y=mx+c)
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50);

#Regression Evaluation Metrics
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('RSquare:', metrics.r2_score(y_test, predictions))