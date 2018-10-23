# coding=utf-8
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

# np.size(a, axis=NOne) a输入矩阵，axis=0指定为行数，为１返回列数
# np.eye() 生成一个对角矩阵, 默认对角为１
# plt.plot(x,y,color,ms)　x,y为数据，color为颜色,ms为大小
# plt.show()展示所画的散点图
# np.loadtxt导入txt文档
# data[:.0]提取一列数据
# np.zeros(shape, dtype=float, order='C')
# shape int or tuple of ints 　
# np.vstack(tup) 参数tup可以是元组，列表，或者是numpy数组，返回结果为numpy的数组，是垂直的将元素堆叠起来
# np.ones(shape, dtype=None, order='C')用法与np.zeros类似，不过初始化为１

# Basic Function
print('Running warmUpExercise')
print('5x5 Identity Matrix: ')
A = np.eye(5)  # 输出对角矩阵，默认对角为１
print(A)


# 可视化数据
def plot_data(x, y):
    plt.plot(x, y, 'rx', ms=10)
    plt.xlabel('Population of City in 10,000')
    plt.ylabel('profit in $10,000')
    plt.show()
# plt.x_label　set x corrend label，　

print('Plottiong Data...')
data = np.loadtxt('ex1data1.txt',delimiter=',')
print(data)
x = data[:, 0]
y = data[:, 1]
print(x)
print(y)
m = np.size(y, 0)
plot_data(x,y)
# data[:. ]取出一列,必须使用numpy进行转换

# 计算损失函数值, using Gradient descent　method
def compute_cost(x, y, theta):
    ly = np.size(y, 0)
    #损失函数公式J(θ)= ∑(h(x)-y)^2/2*size h(x)=theta*x其中dot是矩阵乘 如果两个都为1-D arrays　则为计算内积
    cost = (x.dot(theta)-y).dot(x.dot(theta)-y)/(2*ly)
    return cost

# 迭代计算theta, numiters迭代的个数，创造numiters个tuples初始化为０　array[0,0,0,0...]
# theta为一个矩阵，每次迭代将theta0,theta1都进行计算了，之后同时更新，然后将cost值计算并返回，加逗号主要是因为元组中只有一个元素时将其转换为元组类型
def gradient_descent(x, y, theta, alpha, numiters):
    m = np.size(y, 0)
    j_history = np.zeros((numiters, ))

    for i in range(numiters):
        deltaj = x.T.dot(x.dot(theta)-y)/m
        theta = theta-alpha*deltaj
        j_history[i] = compute_cost(x,y,theta)
    return  theta, j_history

print('Running Gradient Descent...')
x = np.vstack((np.ones((m, )), x)).T # 构造矩阵x的转置用于计算h(theta)  hopythesis function
theta = np.zeros((2,))

iterations = 1500
alpha = 0.01

J =compute_cost(x, y, theta) # 计算出第一次的cost值
print(J)

theta, j_history = gradient_descent(x,y,theta,alpha,iterations)

plt.plot(x[:, 1], y, 'rx', ms=10, label='Training data') # plot y using x as index
plt.plot(x[:, 1], x.dot(theta), '-', label='linear regression')
plt.xlabel('Population of CIty in 10,000')
plt.ylabel('Profit in $10,000')
plt.legend(loc='upper right')
plt.show()

# predict values for populatiob sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]).dot(theta)
print("For population = 35,000, we predict a profit of", predict1*10000)
predict2 = np.array([1, 7.0]).dot(theta)
print("For population = 70,000, we predict a profit of", predict2*10000)

print('Visualizing J(theta_0, theta_1)')
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros((np.size(theta0_vals, 0), np.size(theta1_vals, 0)))

for i in range(np.size(theta0_vals, 0)):
    for j in range(np.size(theta1_vals, 0)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i, j] = compute_cost(x, y, t)
