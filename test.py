# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def logist(z):
    ans = []
    for i in z:
        tem = []
        tem.append(1 / (1 + math.exp(-i[0])))
        ans.append(tem)
    return ans


def computeCost(X, Y, theta):
    sum = 0
    for i in range(len(Y)):
        h = 1 / (1 + math.exp(-np.dot(X.T[i], theta)))
        if h==0:
            if Y[i][0]==0:
                sum+=0
            else:
                sum-=999999
        elif h==1:
            if Y[i][0]==1:
                sum+=0
            else:
                sum-=999999
        else:sum += Y[i][0] * math.log(h) + (1 - Y[i][0]) * math.log(1 - h)
    return sum


def logistic_regression(X, Y, theta, rate=0.001, thredsome=-0.1, maxstep=10000):
    # update theta
    cost = computeCost(X, Y, theta)
    picturelist = []
    picturelist.append(cost)
    step = 0
    while cost<thredsome and step < maxstep:
        tem = theta - rate * np.dot(X, logist(np.dot(X.T, theta)) - Y)
        theta = tem
        cost = computeCost(X, Y, theta)
        picturelist.append(cost)
        step += 1
    # if cost>thredsome:
    #     print("发散")
    return theta, step, picturelist


f = open('spambase.data')
df = pd.read_csv(f, header=None)
df = df.values.tolist()
featurelist = []
for i in df:
    y = i[0].split()
    for j in range(len(y)):
        y[j] = float(y[j])
    featurelist.append(y)

X = []
Y = []

for i in featurelist:
    tem = []
    tem.append(i[-1])
    Y.append(tem)
    X.append(i[0:-1])

Y = np.array(Y)
X = np.array(X)

stand = StandardScaler()
X = stand.fit_transform(X)
X = X.T
X = np.insert(X, 0, [1], axis=0)
# train_X=X[:,0:4000]
# test_X=X[:,4000:]
train_X, test_X, train_Y, test_Y = train_test_split(X.T, Y)

train_X = train_X.T
test_X = test_X.T
print(train_X.shape)
print(test_X.shape)
print(train_Y.shape)
print(test_Y.shape)

theta = np.zeros([len(X), 1], dtype=float)
# cur_theta_BGD, step, lost_BGD = logistic_regression(train_X, train_Y, theta)
#
# picturex_BGD=np.arange(len(lost_BGD))
# ax = plt.subplot(111)
# plt.plot(picturex_BGD,lost_BGD,color='blue')
# plt.title(" the trend of likely function in training")
# ax.set_xlabel("step")
# ax.set_ylabel("value")
# plt.savefig("./likely_function.png")
# plt.show()
#
# print(cur_theta_BGD)
# predict_BGD = logist(np.dot(test_X.T, cur_theta_BGD))
#
# predict_Y = []
# # 预测正确的邮件数
# predict_spam = 0
# predict_good = 0
# # 测试集里的邮件数
# test_good = 0
# test_spam = 0
#
# for i in test_Y:
#     if i[0] == 1:
#         test_spam += 1
#     else:
#         test_good += 1
#
# for i in predict_BGD:
#     if i[0] >= 0.5:
#         predict_Y.append(1)
#     else:
#         predict_Y.append(0)
#
# for i in range(len(predict_Y)):
#     if predict_Y[i] == test_Y[i][0]:
#         if predict_Y[i] == 1:
#             predict_spam += 1
#         else:
#             predict_good += 1
#
# print("测试集中的正常邮件数为", test_good, "预测对的正常邮件数为", predict_good)
# print("测试集中的垃圾邮件数为", test_spam, "预测对的垃圾邮件为", predict_spam)
# print("总预测正确个数为", (predict_spam + predict_good), "总正确率为", (predict_spam + predict_good) / len(predict_Y))
