
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
# 导入数据集
'''
dataset = pd.read_csv('Data.csv')
# 取除最后一列外的所有数据
x = dataset.iloc[:, :-1].values
# print(x)
# 取所第三列的所有数据
y = dataset.iloc[:, 3].values
# print(y)

'''
# 处理缺失值
'''
# from sklearn.preprocessing import Impute #新版sklearn中Imputer已改为SimpleImputer
from sklearn.impute import SimpleImputer

# imputer = Impyter(missing_values='Nan', strategy='mean', axis=0) #方法更改，原视频中的参数已失效
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')



'''
# 数据处理分类数据（按数据类别分1，2，3.....）
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
# print(x)

# 虚拟编码,将国家分组标签转换为数字标签
# 'categorical_features'参数在0.20版中已弃用，在0.22版中删除；需改用ColumnTransformer
# onehotencoder = OneHotEncoder(categories_=[0])
# x=onehotencoder.fit_transform(x).toarray()
# print(x)
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([
    ('Country',  # 只是个名字,可以随意填写;
     OneHotEncoder(),  # 转换器类
     [0]  # [0]为需要处理的为第一列;
     )
], remainder='passthrough'  # remainder='passthrough'表示不丢弃没有处理的数据，默认值'drop'表示丢弃没有经过处理的数据
)
x = ct.fit_transform(x)
# print(x)
# 拟合平均值（分类）
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)
# 编码结果标签
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
# print(y)

'''
# 划分测试集与训练集
'''
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# print(x_train,'\n')
# print(x_test,'\n')
# print(y_train,'\n')
# print(y_test,'\n')

'''
# 特征缩放
'''
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
print(x_train, '\n', x_test)
