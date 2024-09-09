import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge, Lasso


def sigmoid_scaling(X):
    # 将数据映射到 [0, 1] 范围
    return 1 / (1 + np.exp(-X))


# 绘制真实值 vs 预测值
def show_graph(y_test, y_pred, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'{title} True Values vs Predictions')
    plt.legend()
    plt.show()


# 绘制代价函数随迭代次数的变化图像
def plot_cost_history(J_history, title):
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(J_history)), J_history, color='blue')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost (J)')
    plt.title(f'{title} Cost Function History')
    plt.grid(True)
    plt.show()


# 批量梯度下降实现
def batch_gradient_descent(X, y, alpha=0.01, num_iterations=1000, compute_cost=True):
    m, n = X.shape
    theta = np.zeros(n)  # 初始化参数
    J_history = np.zeros(num_iterations)  # 存储每次迭代的代价

    for i in range(num_iterations):
        predictions = X.dot(theta)  # 每个样本在theta上的预测结果
        errors = predictions - y  # 每个样本的损失(h(xi)-yi)
        gradient = (1 / m) * X.T.dot(errors)  # 第i维度theta的梯度=所有样本第i维度值*损失求和(∑xi*(h(xi)-yi))
        theta -= alpha * gradient
        if compute_cost:
            J_history[i] = (1 / (2 * m)) * np.sum(errors ** 2)  # 计算代价函数

    return theta, J_history


# 随机梯度下降实现
def stochastic_gradient_descent(X, y, alpha=0.01, num_iterations=1000, compute_cost=True):
    m, n = X.shape
    theta = np.zeros(n)  # 初始化参数
    J_history = np.zeros(num_iterations)  # 存储每次迭代的代价

    for i in range(num_iterations):
        for j in range(m):
            random_index = np.random.randint(m)  # 随机选取样本
            X_i = X[random_index:random_index + 1]
            y_i = y[random_index:random_index + 1]
            prediction = X_i.dot(theta)  # 该样本下的预测值
            error = prediction - y_i  # 该样本下的误差
            gradient = X_i.T.dot(error)  # 该样本对应的梯度
            theta -= alpha * gradient
        # 一代结束后
        if compute_cost:
            predictions = X.dot(theta)
            errors = predictions - y
            J_history[i] = (1 / (2 * m)) * np.sum(errors ** 2)  # 计算代价函数

    return theta, J_history


def printxishu(theta):
    theta.reshape(-1)
    print(
        f"城市犯罪率:{theta[0]}\n住宅用地比例:{theta[1]}\n城市中非零售业务的占比:{theta[2]}\n是否邻近查尔斯河:{theta[3]}\n一氧化氮浓度（每10亿分之一）:{theta[4]}\n每栋住宅的平均房间数:{theta[5]}\n自1940年以来建成的房屋比例:{theta[6]}\n距离5个波士顿就业中心的加权距离:{theta[7]}\n辐射性公路的可达性指数:{theta[8]}\n每1000美元的物业税率:{theta[9]}\n学生与教师的比例:{theta[10]}\n黑人比例:{theta[11]}\n低收入人口比例:{theta[12]}\n")


# 读取数据
df = pd.read_csv("housing_data.csv", sep=r'\s+')
X = df.drop(columns=['medv']).values
y = df['medv'].values

# 归一化数据
X_scaled = sigmoid_scaling(X)

# 使用批量梯度下降进行训练
o_thetab, o_J_historyb = batch_gradient_descent(X, y, alpha=0.000001, num_iterations=2000)
o_thetas, o_J_historys = stochastic_gradient_descent(X, y, alpha=0.000001, num_iterations=2000)
thetab, J_historyb = batch_gradient_descent(X_scaled, y, alpha=0.01, num_iterations=2500)
thetas, J_historys = stochastic_gradient_descent(X_scaled, y, alpha=0.01, num_iterations=2500)

# 预测结果
y_pred_bgd = X_scaled.dot(thetab)
y_pred_sgd = X_scaled.dot(thetas)
o_y_pred_bgd = X.dot(o_thetab)
o_y_pred_sgd = X.dot(o_thetas)

# 绘制代价函数历史图像
plot_cost_history(J_historyb, "bgd_sigmoid")
plot_cost_history(J_historys, "sgd_sigmoid")
plot_cost_history(o_J_historyb, "bgd_nosigmoid")
plot_cost_history(o_J_historys, "sgd_nosigmoid")

# 绘制真实值 vs 预测值
show_graph(y, y_pred_bgd, "bgd_sigmoid")
show_graph(y, y_pred_sgd, "sgd_sigmoid")
show_graph(y, o_y_pred_bgd, "bgd_nosigmoid")
show_graph(y, o_y_pred_sgd, "sgd_nosigmoid")

# 使用岭回归训练模型
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_scaled, y)
y_pred_ridge = ridge_reg.predict(X_scaled)

# 使用LASSO回归训练模型
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_scaled, y)
y_pred_lasso = lasso_reg.predict(X_scaled)

# 绘制真实值 vs 预测值图像
show_graph(y, y_pred_sgd, 'SGDRegressor')
show_graph(y, y_pred_ridge, 'Ridge Regression')
show_graph(y, y_pred_lasso, 'LASSO Regression')

# 打印和比较模型的系数
print("SGDRegressor Coefficients:")
printxishu(thetas)
print("Ridge Regression Coefficients:")
printxishu(ridge_reg.coef_)
print("LASSO Regression Coefficients:")
printxishu(lasso_reg.coef_)
