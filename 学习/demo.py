import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
#%matplotlib inline
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split  # 对数据集切分
from sklearn.metrics import r2_score

# 机器算法模型
from sklearn.neighbors import KNeighborsRegressor  # KNN，即K近邻算法
from sklearn.linear_model import LinearRegression  # 多元线性回归算法
from sklearn.linear_model import Ridge  # 线性回归算法Ridge回归，岭回归
from sklearn.linear_model import Lasso  # 线性回归算法Lasso回归，可用作特征筛选
from sklearn.tree import DecisionTreeRegressor  # 决策树，既可以做分类也可以做回归（本文主要用于分类问题）
from sklearn.svm import SVR  # 支持向量机

# 生成训练数据和测试数据
#boston = datasets.load_boston()
train, target = datasets.fetch_openml(name="boston", version=1, as_frame=True, return_X_y=True, parser="pandas")

#print(train)
#exit()

#train = boston.data  # 样本
#target = boston.target  # 标签
 
# 切割数据样本集合测试集
X_train, x_test, y_train, y_true = train_test_split(train, target, test_size=0.2)  # 20%测试集；80%训练集
x_test = x_test.astype(np.float64)

# 可视化
data_df = pd.DataFrame(train)
data_df['房价值'] = target
data_df.head(10)

#print(data_df)
#exit()

# 创建学习模型
knn = KNeighborsRegressor()
linear = LinearRegression()
ridge = Ridge()
lasso = Lasso()
decision = DecisionTreeRegressor()
svr = SVR()

# 训练模型
knn.fit(X_train, y_train)  # 学习率、惩罚项都封装好了
linear.fit(X_train, y_train)
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
decision.fit(X_train, y_train)
svr.fit(X_train, y_train)

# 预测数据
y_pre_knn = knn.predict(x_test)
y_pre_linear = linear.predict(x_test)
y_pre_ridge = ridge.predict(x_test)
y_pre_lasso = lasso.predict(x_test)
y_pre_decision = decision.predict(x_test)
y_pre_svr = svr.predict(x_test)
 
print(linear.coef_)  # w值
print(linear.intercept_)  # b值

# 评分，R2 决定系数（拟合优度）。模型越好：r2→1；模型越差：r2→0
knn_score = r2_score(y_true, y_pre_knn)
linear_score = r2_score(y_true, y_pre_linear)
ridge_score = r2_score(y_true, y_pre_ridge)
lasso_score = r2_score(y_true, y_pre_lasso)
decision_score = r2_score(y_true, y_pre_decision)
svr_score = r2_score(y_true, y_pre_svr)

# 绘图
# KNN
plt.plot(y_true, label='true')
plt.plot(y_pre_knn, label='knn')
plt.legend()
 
# Linear
plt.plot(y_true, label='true')
plt.plot(y_pre_linear, label='linear')
plt.legend()
 
# Ridge
plt.plot(y_true, label='true')
plt.plot(y_pre_ridge, label='ridge')
plt.legend()
 
# Lasso
plt.plot(y_true, label='true')
plt.plot(y_pre_lasso, label='lasso')
plt.legend()
 
# Decision
plt.plot(y_true, label='true')
plt.plot(y_pre_decision, label='decision')
plt.legend()
 
# SVR
plt.plot(y_true, label='true')
plt.plot(y_pre_svr, label='svr')
plt.legend()
 
plt.show()