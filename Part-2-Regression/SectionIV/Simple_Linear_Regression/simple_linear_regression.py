# Simple Linear Regression 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# 划分训练集与测试集
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 3, random_state=0)

# print(x_train, '\n')
# print(x_test, '\n')
# print(y_train, '\n')
# print(y_test, '\n')

# 创建拟合回归器
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
# print(y_pred)

# 画图
# 训练集结果
plt.scatter(x_train,y_train,color='red')
plt.scatter(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary VS Experience(training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
# 测试机结果
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,regressor.predict(x_test),color='blue')
plt.title('Salary VS Experience(test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

