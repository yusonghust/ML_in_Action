# -*- coding:utf-8 -*-

'''
实现一个简单的基于SMO的SVM
'''

import numpy as np

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
        j = random.choice(range(m))
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
    alphas = np.mat(np.zeros(m,1))

    iter = 0
    while (iter < maxIter):
        # 记录alpha是否已经进行优化，每次循环时设为0，然后再对整个集合顺序遍历
        alphaPairsChanged = 0
        for i in range(m):

            fx_i = float(np.multiply(alphas,labelMat).T*(dataMat*dataMat[i,:].T))
            #计算预测值与真实值的误差
            E_i = fx_i - float(labelMat[i])

            # 检验训练样本(xi, yi)是否满足KKT条件
            # yi*f(i) >= 1 and alpha = 0 (outside the boundary)
            # yi*f(i) == 1 and 0<alpha< C (on the boundary)
            # yi*f(i) <= 1 and alpha = C (between the boundary)
            # labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了。
            #
