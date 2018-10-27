import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as opt
# Load Data
data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, 0:2]
Y = data[:, 2]

# ==================== Part 1: Plotting ====================
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

# 绘制散点图像
def plotData(x, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    p1 = plt.scatter(x[pos, 0], x[pos, 1], marker='+', s=30, color='b')
    p2 = plt.scatter(x[neg, 0], x[neg, 1], marker='o', s=30, color='y')
    plt.legend((p1, p2), ('Admitted', 'Not admitted'), loc='upper right', fontsize=8)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.show()

plotData(X, Y)

# ============ Part 2: Compute Cost and Gradient ============
m, n = np.shape(X)
X = np.concatenate((np.ones((m, 1)), X), axis=1)
init_theta = np.zeros((n+1,))

# sigmoid函数
def sigmoid(z):
    g = 1/(1+np.exp(-1*z))
    return g

# 计算损失函数和梯度函数
def costFunction(theta, x, y):
    m = np.size(y, 0)
    h = sigmoid(x.dot(theta))
    if np.sum(1-h < 1e-10) != 0:
        return np.inf
    j = -1/m*(y.dot(np.log(h))+(1-y).dot(np.log(1-h)))
    return j

def gradFunction(theta, x, y):
    m = np.size(y, 0)
    grad = 1 / m * (x.T.dot(sigmoid(x.dot(theta)) - y))
    return grad


cost = costFunction(init_theta, X, Y)
grad = gradFunction(init_theta, X, Y)
print('Cost at initial theta (zeros): ', cost)
print('Gradient at initial theta (zeros): ', grad)

result = opt.minimize(costFunction, x0=init_theta,method='BFGS',jac=gradFunction,args=(X, Y))
#print(result)
theta = result.x
print('Cost at theta found by fmin_bfgs', result.fun)
print('plt_x is ',np.array([np.min(X[:, 1])-2,np.max(X[:, 1])+2]))
#=====================绘制边界图像======================
def plotDecisionBoundary(theta, x, y):
    pos = np.where(y==1)
    neg = np.where(y==0)
    p1 = plt.scatter(x[pos, 1], x[pos, 2], marker='+', s=60, color='r')
    p2 = plt.scatter(x[neg, 1], x[neg, 2], marker='o', s=30, color='y')
    plt_x = np.array([np.min(x[:, 1])-2,np.max(x[:, 1])+2]) # 用于绘制边界
    plt_y = -1 / theta[2]*(theta[1]*plt_x+theta[0])
    plt.plot(plt_x,plt_y)
    plt.legend((p1,p2),('Admitted','Not admitted'), loc='upper right', fontsize=8)
    plt.xlabel('Exam 1 score')
    plt.ylabel('exam 2 score')
    plt.show()

plotDecisionBoundary(theta,X,Y)

#================预测===========
def predict(theta, x):
    probability = sigmoid(x.dot(theta))
    return [1 if x >= 0.5 else 0 for x in probability]
p = predict(theta,X)
print('Train Accuracy: ',np.sum(p == Y)/np.size(Y, 0))
print(p)