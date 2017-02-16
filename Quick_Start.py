# -*-coding:utf-8-*-
'''
    Author：kevinelstri
    Datetime:2017.2.16
'''
# -----------------------
# An introduction to machine learning with scikit-learn
# http://scikit-learn.org/stable/tutorial/basic/tutorial.html
# -----------------------

'''
    通过使用sklearn，简要介绍机器学习，并给出一个简单的例子
'''

'''
    Machine learning: the problem setting
'''

'''
    机器学习：
        就是对数据的一系列样本进行分析，来预测数据的未知结果。
    监督学习：
        数据的预测来自于对已有的数据进行分析，进而对新增的数据进行预测。
        监督学习可以划分为两类：分类和回归
    非监督学习：
        训练数据由一系列没有标签的数据构成，目的就是发现这组数据中的相似性，也称作聚类。
        或者来发现数据的分布情况，称为密度估计。

    训练数据集、测试数据集：
        机器学习就是通过学习一组数据，来将结果应用于一组新的数据中。
        将一组数据划分为两个集合，一个称为训练集，一个称为测试集。
'''

'''
    Loading an example dataset
'''

'''
    sklearn 有一些标准的数据集，iris,digits 数据集用于分类，boston house prices 数据集用于回归
'''
from sklearn import datasets

iris = datasets.load_iris()  # 加载iris数据集
digits = datasets.load_digits()  # 加载digits数据集
# print iris
# print digits
'''
    数据集是一个类似字典的对象，它保存所有的数据和一些有关数据的元数据。
    数据存储在.data中，这是一个(n_sample, n_features)数组。
    在监督学习问题中，多个变量存储在.target中。
    data:数据
    target:标签

    n_sample:样本数量
    n_features:预测结果的数量
'''

print 'digits.data:', digits.data  # 用来分类样本的特征
print 'digits.target:', digits.target  # 给出了digits数据集的真实值，就是每个数字图案对应的想预测的真实数字

print 'iris.data:', iris.data
print 'iris.target:', iris.target

print digits.images[0]
print digits.images

'''
    Recognizing hand-written digits
'''
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics

digits = datasets.load_digits()  # 加载数据集
'''
    digits数据集中每一个数据都是一个8*8的矩阵
'''
images_and_labels = list(zip(digits.images, digits.target))  # 每个数据集都与标签对应，使用zip()函数构成字典
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training:%i' % label)

n_samples = len(digits.images)  # 样本的数量
print n_samples
data = digits.images.reshape((n_samples, -1))

classifier = svm.SVC(gamma=0.001)  # svm预测器

classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])  # 使用数据集的一半进行训练数据

expected = digits.target[n_samples / 2:]
predicted = classifier.predict(data[n_samples / 2:])  # 预测剩余的数据

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))  # 图片与预测结果按照字典方式对应
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')  # 展示图片
    plt.title('Prediction: %i' % prediction)  # 标题

# plt.show()

'''
    Learning and predicting
'''
'''
    digits数据集，就是给定一个图案，预测其表示的数字是什么。
    样本共有10个可能的分类（0-9），通过匹配(fit)预测器(estimator)来预测(predict)未知样本所属的分类。
    sklearn中，分类的预测器就是为了实现fit(X,y)和predict(T)两个方法（匹配和预测）。

    fit(X,y):训练数据
    predict(T):预测数据

    预测器sklearn.svm.SVC,就是为了实现支持向量机分类
'''
from sklearn import svm

clf = svm.SVC(gamma=0.001, C=100.)
'''
    预测器的名字是clf,这是一个分类器，它必须进行模型匹配(fit)，也就是说，必须从模型中学习。
    从模型学习的过程，模型匹配的过程，是通过将训练集传递给fit方法来实现的。

    本次实验中将除了最后一个样本的数据全部作为训练集[:-1]
'''
print clf.fit(digits.data[:-1], digits.target[:-1])  # 对前面所有的数据进行训练
print clf.predict(digits.data[-1:])  # 对最后一个数据进行预测

'''
    Model persistence
        使用pickle保存训练过的模型
'''
from sklearn import svm
from sklearn import datasets

clf = svm.SVC()  # 构造预测器
iris = datasets.load_iris()  # 加载数据集
X, y = iris.data, iris.target  # 数据的样本数和结果数
clf.fit(X, y)  # 训练数据

import pickle

s = pickle.dumps(clf)  # 保存训练模型
clf2 = pickle.loads(s)  # 加载训练模型
print clf2.predict(X[0:1])  # 应用训练模型

# 在scikit下，可以使用joblib's(joblib.dump, joblib.load)来代替pickle
from sklearn.externals import joblib

joblib.dump(clf, 'filename.pkl')  # 保存训练模型
clf = joblib.load('filename.pkl')  # 加载数据模型
print clf.predict(X[0:1])  # 应用训练模型

'''
    Conventions
'''
from sklearn import datasets
from sklearn.svm import SVC

iris = datasets.load_iris()
clf = SVC()
clf.fit(iris.data, iris.target)
print list(clf.predict(iris.data[:3]))  # output:[0,0,0]
# 由于iris.target是整型数组，所以这里的predict()返回的也是整型数组

clf.fit(iris.data, iris.target_names[iris.target])
print list(clf.predict(iris.data[:3]))  # output:['setosa', 'setosa', 'setosa']
# 这里iris.target_names是字符串名字，所以predict()返回的也是字符串

'''
    Refitting and updating parameters
'''
import numpy as np
from sklearn.svm import SVC

rng = np.random.RandomState(0)
X = rng.rand(100, 10)
y = rng.binomial(1, 0.5, 100)
X_test = rng.rand(5, 10)
clf = SVC()
clf.set_params(kernel='linear').fit(X, y)
print clf.predict(X_test)  # output:[1, 0, 1, 1, 0]

clf.set_params(kernel='rbf').fit(X, y)
print clf.predict(X_test)  # output:[0, 0, 0, 1, 0]


'''
    Multiclass vs. multilabel fitting
'''
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]

classif = OneVsRestClassifier(estimator=SVC(random_state=0))
print classif.fit(X, y).predict(X)  # output:[0 0 1 1 2]

y = LabelBinarizer().fit_transform(y)
print classif.fit(X, y).predict(X)  # output:[[1 0 0][1 0 0][0 1 0][0 0 0][0 0 0]]

from sklearn.preprocessing import MultiLabelBinarizer
y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
y = MultiLabelBinarizer().fit_transform(y)
print classif.fit(X, y).predict(X)  # output:[[1 1 0 0 0][1 0 1 0 0][0 1 0 1 0][1 0 1 0 0][1 0 1 0 0]]
























