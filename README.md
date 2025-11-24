# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the needed packages. 
2. Assigning hours to x and scores to y.
3. Plot the scatter plot.
4. Use mse,rmse,mae formula to find the values.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Thamizh S
RegisterNumber: 212224040350
*/
```
```python
# IMPORT REQUIRED PACKAGE
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
# READ CSV FILES
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
# COMPARE DATASET
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
# PRINT PREDICTED VALUE
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
# GRAPH PLOT FOR TRAINING SET
print("\nNAME:Thamizh.S")
print("REG NO:21224040350")
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,reg.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# GRAPH PLOT FOR TESTING SET
print("\nNAME:Thamizh.S")
print("REG NO:21224040350")
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# PRINT THE ERROR
print("NAME:Thamizh.S")
print("REG NO:21224040350")
mse=mean_absolute_error(y_test,y_pred)
print('\nMean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

```

## Output:

### To Read Head and Tail Files:

<img width="228" height="200" alt="image" src="https://github.com/user-attachments/assets/502dd37b-c51f-409e-995d-82898ad6cfba" />

<img width="179" height="206" alt="image" src="https://github.com/user-attachments/assets/22e8f4a5-0c79-4dec-bf96-30f6ca1f7203" />


Compare Dataset

<img width="196" height="530" alt="image" src="https://github.com/user-attachments/assets/6967c43b-4685-4d33-9c8e-a3350889545d" />


Predicted Value

<img width="701" height="195" alt="image" src="https://github.com/user-attachments/assets/d430cf85-f980-4286-8d8c-9db2db5307d2" />



Graph For Training Set
<img width="744" height="628" alt="image" src="https://github.com/user-attachments/assets/32b13a72-1a69-4144-af4b-b2d901cd51f9" />



Graph For Testing Set

<img width="749" height="604" alt="image" src="https://github.com/user-attachments/assets/fb1d5541-3dbf-474c-83d7-1c7db3322588" />


Error

<img width="308" height="156" alt="image" src="https://github.com/user-attachments/assets/898f18ba-ed7b-4a6a-95de-7a0dc6531910" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
