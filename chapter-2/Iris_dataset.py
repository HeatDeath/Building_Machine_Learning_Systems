from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
features = data['data']
feature_names = data['feature_names']
target = data['target']
labels = data['target_names'][data['target']]

# # 四种特征
# print(features)
# print('---------------------------------------')
# # 四种特征的名称
# print(feature_names)
# # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# print('---------------------------------------')
# # 样本的实际分类
# print(target)
# print('---------------------------------------')


for t, marker, c in zip(range(3), '>ox', 'rgb'):
    # features[target == t,0] 代表 样本分类 为 t 的样本的 第一个特征
    plt.scatter(features[target == t,0],
                features[target == t,1],
                marker=marker,
                c=c)
# plt.show()

plength = features[:, 2]
is_setosa = (labels == 'setosa')