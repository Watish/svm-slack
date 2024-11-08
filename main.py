import numpy as np
import scipy.io as sio
import random
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# 加载数据
path = "E:\\py\\data.mat"
mat = sio.loadmat(path)
dataset = mat['y']

# 提取特征和标签
X = dataset[:, :2]
y = dataset[:, 2]

# 生成线性不可分的数据点
np.random.seed(0)  # 设置随机种子以保证结果可复现
n_outliers = 5  # 异常点的数量
outliers_X = np.random.uniform(X.min(), X.max(), (n_outliers, 2))
outliers_y = np.array([random.choice([-1, 1]) for _ in range(n_outliers)])  # 随机选择类别标签

# 合并原始数据与新生成的异常点
X = np.vstack((X, outliers_X))
y = np.hstack((y, outliers_y))

# 创建SVM模型，使用线性核函数并设置松弛变量C
svm = SVC(kernel='linear', C=0.01)  # 调整C值以控制松弛变量的影响程度
#svm = SVC(kernel='linear', C=1.0)
# 训练模型
svm.fit(X, y)

# 预测
predictions = svm.predict(X)

# 计算准确率
accuracy = np.mean(predictions == y)
print(f"Accuracy after adding outliers and retraining: {accuracy * 100:.2f}%")

# 绘制结果
fig = plt.figure()

# 绘制数据点
x1 = X[y == 1, 0]
y1 = X[y == 1, 1]
x2 = X[y == -1, 0]
y2 = X[y == -1, 1]
plt.scatter(x1, y1, c='r', label='Class 1')
plt.scatter(x2, y2, c='b', label='Class -1')

# 绘制决策边界
w = svm.coef_[0]
b = svm.intercept_[0]
x = np.arange(X[:, 0].min(), X[:, 0].max(), 0.1)
y = (-w[0] / w[1]) * x - b / w[1]
plt.plot(x, y, c='g', label='Decision Boundary')

# 绘制支持向量
sv = svm.support_vectors_
plt.scatter(sv[:, 0], sv[:, 1], c='k', marker='x', label='Support Vectors')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Linear SVM Classification with Outliers')
plt.show()