# -*- coding:utf-8 -*-

'''
实现一个简单的基于SMO的SVM
'''

import numpy as np
import matplotlib.pyplot as plt
import random

def loadDataSet(fileName):
    """
    对文件进行逐行解析，从而得到第行的类标签和整个特征矩阵
    Args:
        fileName 文件名
    Returns:
        dataMat  特征矩阵
        labelMat 类标签
    """
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i, m):
    """
    随机选择一个整数
    Args:
        i  第一个alpha的下标
        m  所有alpha的数目
    Returns:
        j  返回一个不为i的随机数，在0~m之间的整数值
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    """clipAlpha(调整aj的值，使aj处于 L<=aj<=H)
    Args:
        aj  目标值
        H   最大值
        L   最小值
    Returns:
        aj  目标值
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def SMO_simple(data,label,C,toler,maxIter):
    '''
    Args:
    data: 数据集
    label: 数据标签
    toler: 容错率
    C: 松弛变量
    maxIter: 最大循环次数

    Returns:
    b: 模型的常量
    alphas 拉格朗日乘子

    '''

    dataMat = np.mat(data)
    labelMat = np.mat(label).transpose()

    m,n = np.shape(dataMat)

    #初始化参数alphas与b
    b = 0
    alphas = np.zeros((m,1))

    iter = 0
    while (iter < maxIter):
        # 记录alpha是否已经进行优化，每次循环时设为0，然后再对整个集合顺序遍历
        alphaPairsChanged = 0
        for i in range(m):

            fx_i = float(np.multiply(alphas,labelMat).T*(dataMat*dataMat[i,:].T)) + b
            #计算预测值与真实值的误差
            E_i = fx_i - float(labelMat[i])

            # 检验训练样本(xi, yi)是否满足KKT条件
            # yi*f(i) >= 1 and alpha = 0 (outside the boundary)
            # yi*f(i) == 1 and 0<alpha< C (on the boundary)
            # yi*f(i) <= 1 and alpha = C (between the boundary)
            # labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了。
            if(((labelMat[i]*E_i < -toler) and (alphas[i]<C)) or ((labelMat[i]*E_i > toler) and (alphas[i] > 0))):
                # 如果满足优化的条件，我们就随机选取非i的一个点，进行优化比较
                j = selectJrand(i, m)
                # 预测j的结果
                fx_j = float(np.multiply(alphas,labelMat).T*(dataMat*dataMat[j,:].T)) + b
                #计算预测值与真实值的误差
                E_j = fx_j - float(labelMat[j])
                #为了方便更新之后检测参数的变化，首先copy一下,这样后面的改动不会影响这里的结果
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接执行continue语句
                # labelMat[i] != labelMat[j] 表示异侧，就相减，否则是同侧，就相加。
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                # 如果相同，就没发优化了
                if L == H:
                    continue

                # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
                # 参考《统计学习方法》李航-P125~P128<序列最小最优化算法>
                eta = 2.0 * dataMat[i, :]*dataMat[j, :].T - dataMat[i, :]*dataMat[i, :].T - dataMat[j, :]*dataMat[j, :].T
                if eta >= 0:
                    continue

                #更新alpha
                alphas[j] = alphas[j] - labelMat[j]*(E_i-E_j)/eta
                #限制其更新范围
                alphas[j] = clipAlpha(alphas[j],H,L)
                #检测参数更新的幅度，如果更新幅度太小直接退出循环
                if(abs(alphas[j] - alphaJold) < 0.00001):
                    continue
                alphas[i] = alphas[i] + labelMat[j]*labelMat[i]*(alphaJold-alphas[j])

                #计算参数b
                b_1_new = b - E_i - labelMat[i]*(alphas[i]-alphaIold)*dataMat[i,:]*dataMat[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMat[i, :]*dataMat[j, :].T
                b_2_new = b - E_j - labelMat[i]*(alphas[i]-alphaIold)*dataMat[i,:]*dataMat[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMat[j, :]*dataMat[j, :].T

                # if((alphas[i]==0) or (alphas[j]==0) or (alphas[i]==C) or (alphas[j]==1)):
                #     b = (b_1_new + b_2_new)/2.0
                if((0<alphas[i]) and (C>alphas[i])):
                    b = b_1_new
                elif((0<alphas[j]) and (C>alphas[j])):
                    b = b_2_new
                else:
                    b = (b_1_new + b_2_new)/2.0

                alphaPairsChanged += 1
        #连续若干次轮后都无法更新参数说明已经收敛
        if(alphaPairsChanged == 0):
            iter = iter +1
        else:
            iter = 0
    return b, alphas

def calcW(alphas,data,label):
    '''
    基于alpha计算w值
    Args:
    alphas:拉格朗日乘子
    data: 数据集
    label: 数据集对应的标签
    Ruturns:
    w 参数
    '''
    dataMat = np.mat(data)
    labelMat = np.mat(label).transpose()
    m,n = np.shape(dataMat)

    #初始化w
    w = np.zeros((n,1))
    for i in range(m):
        w = w + np.multiply(alphas[i]*labelMat[i],dataMat[i,:].T)
    return w


def plotfig_SVM(xMat, yMat, ws, b, alphas):
    """
    参考地址：
       http://blog.csdn.net/maoersong/article/details/24315633
       http://www.cnblogs.com/JustForCS/p/5283489.html
       http://blog.csdn.net/kkxgx/article/details/6951959
    """

    xMat = np.mat(xMat)
    yMat = np.mat(yMat)

    # b原来是矩阵，先转为数组类型后其数组大小为（1,1），所以后面加[0]，变为(1,)
    b = np.array(b)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 注意flatten的用法
    ax.scatter(xMat[:, 0].flatten().A[0], xMat[:, 1].flatten().A[0])

    # x最大值，最小值根据原数据集dataArr[:, 0]的大小而定
    x = np.arange(-1.0, 10.0, 0.1)

    # 根据x.w + b = 0 得到，其式子展开为w0.x1 + w1.x2 + b = 0, x2就是y值
    y = (-b-ws[0, 0]*x)/ws[1, 0]
    ax.plot(x, y)

    for i in range(np.shape(yMat[0, :])[1]):
        if yMat[0, i] > 0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'cx')
        else:
            ax.plot(xMat[i, 0], xMat[i, 1], 'kp')

    # 找到支持向量，并在图中标红
    for i in range(100):
        if alphas[i] > 0.0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'ro')
    plt.show()

def main():
    data,label = loadDataSet('D:/Github/ML_in_Action/ML-in-Action/MachineLearning-dev/input/6.SVM/testSet.txt')

    b, alphas = SMO_simple(data,label,0.6,0.001,40)

    ws = calcW(alphas,data,label)

    plotfig_SVM(data,label,ws,b,alphas)

if __name__ == '__main__':
    main()





































