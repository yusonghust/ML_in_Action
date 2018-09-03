# -*- coding: UTF-8 -*-

import numpy as np
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def createDataSet():
    #读取数据
    data=[]
    label=[]
    #raw data
    # 1.5 50 thin
    # 1.5 60 fat
    # 1.6 40 thin
    # 1.6 60 fat
    # 1.7 60 thin
    # 1.7 80 fat
    # 1.8 60 thin
    # 1.8 90 fat
    # 1.9 70 thin
    # 1.9 80 thin
    with open("/Users/yusong/Documents/python code/ML-in-Action/DicisionTree/data.txt") as f:
        for line in f:
            #特征：身高，体重 标签：胖瘦
            z=line.strip().split(' ')
            data.append([float(i) for i in z[:-1]])
            label.append(z[-1])

    x=np.array(data)
    label=np.array(label)
    #y为0-1形式的标签
    y=np.zeros(label.shape)

    y[label=='fat'] = 1
    print('data:',data,'--------',x,'-------','label:',label,'-------',y)

    return x,y


def train(x_train,y_train):
    '''
    使用信息熵作为划分标准，进行训练
    '''
    clf=tree.DecisionTreeClassifier(criterion='entropy')

    clf.fit(x_train,y_train)
    ''' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''
    print('feature_importance: %s' % clf.feature_importances_)

    #打印预测结果
    y_pre=clf.predict(x_train)

    print('training dataset true label:',y_train)
    print('training dataset pred label:',y_pre)

    return clf


# def show_tree(clf):
#     '''
#     可视化输出
#     把决策树结构写入文件: http://sklearn.lzjqsdd.com/modules/tree.html

#     Mac报错：pydotplus.graphviz.InvocationException: GraphViz's executables not found
#     解决方案：sudo brew install graphviz
#     参考写入： http://www.jianshu.com/p/59b510bafb4d
#     '''
#     # with open("testResult/tree.dot", 'w') as f:
#     #     from sklearn.externals.six import StringIO
#     #     tree.export_graphviz(clf, out_file=f)

#     import pydotplus
#     from sklearn.externals.six import StringIO
#     dot_data = StringIO()
#     tree.export_graphviz(clf, out_file=dot_data)
#     graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

def test(x_test,clf):

    y_pre=clf.predict(x_test)
    y_pre=np.array(y_pre)
    return y_pre

def evaluate_dicision_tree(x_test,y_test,y_pre,clf):

    precision,recall,thresholds = precision_recall_curve(y_test,y_pre)

    answer = clf.predict_proba(x_test)[:, 1]

    target_names = ['thin', 'fat']
    print(classification_report(y_test, answer, target_names=target_names))
    print(answer)

    # print('precision:',precision,'-------','recall:',recall,'--------','thresholds:',thresholds)

def main():

    x,y=createDataSet()

    #拆分数据集，70%作为训练，30%作为测试集
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

    clf=train(x_train,y_train)

    y_pre=test(x_test,clf)

    evaluate_dicision_tree(x_test,y_test,y_pre,clf)

    #可视化输出
    # show_tree(clf)

    return 0

if __name__ == '__main__':
    main()


































