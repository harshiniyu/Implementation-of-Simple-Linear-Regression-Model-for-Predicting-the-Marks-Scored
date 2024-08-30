# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1 Import the standard Libraries. 

2. Set variables for assigning dataset values.

3. Import linear regression from sklearn. 

 4. Assign the points for representing in the graph. 

5. Predict the regression for marks by using the representation of the graph. 

6. Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Harshini Y
RegisterNumber:  212223240050
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

#spilitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred

Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
![image](https://github.com/user-attachments/assets/ba34ac79-9bfb-40bc-80cb-744c6231dcc9)

![image](https://github.com/user-attachments/assets/399c129a-6c12-402d-af08-7f9aad5d90ed)

![image](https://github.com/user-attachments/assets/5d813403-a4d9-449f-bd62-bfda48a64084)

![image](https://github.com/user-attachments/assets/e4af45fd-6680-4ef7-a918-4d592ced24e1)

![image](https://github.com/user-attachments/assets/0dd252f4-66df-4b87-9126-453bed6fcb85)

![image](https://github.com/user-attachments/assets/5957cd94-ef8f-4162-9547-f07bbad05165)

![image](https://github.com/user-attachments/assets/07a7dddd-d3b8-4edb-a19c-b68098d84de1)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
