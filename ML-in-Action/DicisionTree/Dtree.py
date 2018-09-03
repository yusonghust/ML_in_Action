# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 08:50:14 2018

@author: yu
"""
import operator
from math import log
import decisionTreePlot as dtPlot
from collections import Counter
import copy

def loadData():
    # 加载隐形眼镜相关的 文本文件 数据
    fr = open('D:/ML-in-Action/DicisionTree/lenses.txt')
    # 解析数据，获得 features 数据
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 得到数据的对应的 Labels
    features = ['age', 'prescript', 'astigmatic', 'tearRate']
    return lenses,features

def majoritylabel(label):
    result=Counter(label).most_common(1)[0][0]
    return result

def CalcEntropy(DataSet):
    """
    Desc：
        calculate Shannon entropy -- 计算给定数据集的香农熵
    Args:
        DataSet -- 数据集
    Returns:
        shannonEnt -- 返回 每一组 feature 下的某个分类下，香农熵的信息期望
    """
    #统计每个label出现的次数
    label_count = Counter(item[-1] for item in DataSet)
    #计算每个label出现的概率
    probs=[p[1]/len(DataSet) for p in label_count.items()]
    #计算香农熵
    shannonEnt=sum([-p*log(p,2) for p in probs])
    return shannonEnt
    
def splitDataSet(DataSet,index,value):
    """
    Desc：
        划分数据集
        splitDataSet(通过遍历dataSet数据集，求出index对应的colnum列的值为value的行)
        就是依据index列进行分类，如果index列的数据等于 value的时候，就要将 index 划分到我们创建的新的数据集中
    Args:
        DataSet  -- 数据集                 待划分的数据集
        index -- 表示每一行的index列        划分数据集的特征
        value -- 表示index列对应的value值   需要返回的特征的值。
    Returns:
        index 列为 value 的数据集【该数据集需要排除index列】
    """
    retDataSet = [data[:index] + data[index+1:] for data in DataSet for i, v in enumerate(data) if i == index and v == value]
    return retDataSet
    
def chooseBestFeature(DataSet):
    """
    Desc:
        选择切分数据集的最佳特征
    Args:
        DataSet -- 需要切分的数据集
    Returns:
        bestFeatureIndex -- 切分数据集的最优的特征的列序号
    """
    #最后一列是戴眼镜的结果，所以特征数要减一
    featureNum=len(DataSet[0])-1
    #未分裂前的信息熵
    baseEntropy=CalcEntropy(DataSet)
    # 初始化最优的信息增益值, 和最优的Feature编号
    best_info_gain=0
    bestFeatureIndex=-1
    # 遍历每一个特征
    for i in range(featureNum):
        #统计第i个特征
        feature_count=Counter([item[i] for item in DataSet])
        #计算分裂之后的香农熵
        new_entropy=sum([feature[1]/float(len(DataSet)) * CalcEntropy(splitDataSet(DataSet,i,feature[0])) for feature in feature_count.items()])
        
        #计算信息增益
        info_gain=baseEntropy-new_entropy
        if(info_gain>best_info_gain):
            best_info_gain=info_gain
            bestFeatureIndex=i
    return bestFeatureIndex

        
def CreateTree(DataSet,Feature):
    """
    Desc:
        创建决策树
    Args:
        DataSet -- 要创建决策树的训练数据集,最后一列是决策结果，即带什么样的眼镜
        feature -- 训练数据集中特征对应的含义，不是目标变量
    Returns:
        myTree -- 创建完成的决策树
    """
    feature=copy.copy(Feature)
    label=[item[-1] for item in DataSet]
    # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，就只直接返回结果就行
    # 第一个停止条件：所有的类标签完全相同，则直接返回该类标签。
    # count() 函数是统计括号中的值在list中出现的次数
    if(label.count(label[0])==len(label)):
        return label[0]
    
    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    if(len(DataSet[0])==1):
        result=majoritylabel(label)
        return result
    
    # 选择最优的特征分裂，创建子树，得到最优列对应的label含义
    bestFeatureIndex=chooseBestFeature(DataSet)
    FeatureName=feature[bestFeatureIndex]
    
    #初始化子树
    Dtree={FeatureName: {}}
    #去掉这个特征
    feature.pop(bestFeatureIndex)
    #取出最优特征对应的列，分裂
    FeatureValue=[item[bestFeatureIndex] for item in DataSet]
    UniqueValue=set(FeatureValue)
    
    for value in UniqueValue:
        # 求出剩余的特征
        subfeature=feature[:]
        # 遍历包含当前特征值的item，递归调用CreateTree()
        Dtree[FeatureName][value]=CreateTree(splitDataSet(DataSet,bestFeatureIndex,value),subfeature)
        
    return Dtree

def main():
    lenses,features=loadData()
    lenseTree=CreateTree(lenses,features)
    dtPlot.createPlot(lenseTree)
        
if __name__=="__main__":
    main()      
        
        
        
        
        