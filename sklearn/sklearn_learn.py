#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
模块说明：sklearn练习
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics


class MyKnn:
    """KNN最近邻算法:在训练集合中查找距离最近的K和样本，标记新样本的类别的这K个样本中类别最多的类别"""
    def model(self):
        iris = load_iris()
        x = iris.data
        y = iris.target

        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.4)
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(train_x, train_y)
        y_ = clf.predict(test_x)
        proba = clf.predict_proba(test_x)

        # 准确率
        accu = metrics.accuracy_score(y_, test_y)
        print("KNN 预测准确率: ", accu)


class MyLinearModel:
    """线性回归算法实现：基本线性回归、Ridge回归（L1正则化），Lasso回归（L2正则化）"""
    def model(self):
        iris = load_iris()
        x, y = iris.data, iris.target
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.4)

        clf = linear_model.LinearRegression()
        clf.fit(train_x, train_y)

        # w参数
        print(clf.coef_)

        y_ = clf.predict(test_x)
        print('均方误差:', metrics.mean_squared_error(test_y, y_))
        print('R2:', metrics.r2_score(test_y, y_))

    def ridge_mode(self):
        x = 1.0 / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
        y = np.ones([10])

        alphas = np.logspace(-10, -2, 200)
        coefs = []
        for a in alphas:
            # clf = linear_model.Ridge(alpha=a, fit_intercept=False)
            clf = linear_model.Lasso(alpha=a, fit_intercept=False)
            clf.fit(x, y)
            coefs.append(clf.coef_)
        print(coefs)
        ax = plt.gca()
        ax.set_color_cycle(['b','r','g','c','k','y','m'])

        ax.plot(alphas, coefs)
        ax.set_xscale('log')
        ax.set_xlim(ax.get_xlim()[::-1])
        plt.xlabel('alpha')
        plt.ylabel('weights')
        plt.title('Ridge conefficients as a finction')
        plt.axis('tight')
        plt.show()


if __name__ == "__main__":
    # knn = MyKnn()
    # knn.model()

    lr = MyLinearModel()
    lr.ridge_mode()