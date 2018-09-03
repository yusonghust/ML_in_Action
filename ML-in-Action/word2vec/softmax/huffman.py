import math
from collections import Counter
import warnings
import fenci
import numpy as np


#using hierarchical softmax
#节点类
class HuffmanNode():

    def __init__(self,
                 name=None,
                 freq=None #节点出现的次数或者频率
                 ):
        self.name=name
        self.freq=freq
        self.father=None
        self.left_child=None
        self.right_child=None
        self.code=None #节点的编码

    def isLeft(self):
        return self.father.left_child==self

#创建叶子节点,输入为dict,形式为：{name:freq}
def LeafNode(corpus):
    return [HuffmanNode(item[0],item[1]) for item in corpus.items()]




#创建huffman树以及huffman编码
class Huffman():

    def __init__(self):
        self.nodes=[]
        self.root=None


    def bulid_tree(self,nodes):
        #规定分到左边的是负类，分到右边的是正类
        #The rule is divided into the negative class on the left and the positive class on the right.
        queue = nodes
        tree_nodes = []
        k = 0
        while(len(queue)>1):
            queue.sort(key=lambda item: item.freq)
            right_node = queue.pop(0)
            left_node = queue.pop(0)
            freq=right_node.freq+left_node.freq
            father_node = HuffmanNode(k,freq)
            father_node.left_child = left_node
            father_node.right_child = right_node
            left_node.father = father_node
            right_node.father = father_node
            queue.append(father_node)
            tree_nodes.append(right_node)
            tree_nodes.append(left_node)
            k=k+1
        self.root=queue[0]
        tree_nodes.append(queue[0])
        return tree_nodes

    def huffmancoding(self,tree_nodes):
        for i in range(len(tree_nodes)):
            temp = tree_nodes[i]
            code = ''
            while (temp!= self.root):
                if(temp.isLeft()):
                    code = '0' + code
                else:
                    code = '1' + code
                temp = temp.father
            tree_nodes[i].code = code
            self.nodes.append(tree_nodes[i])
        return



#huffman解码,输入的是一棵huffman树，以及需要解码的编码，返回编码对应的节点名称
def huffmandecoding(huffmantree,strs):
    root=huffmantree.root #从根节点开始解码
    for i in strs:
        if(i=='0'):
            root=root.left_child
        else:
            root=root.right_child
    if(root==None):
        return 'encoding error'
    else:
        print('--------huffman code is:',strs,'--------huffman node is:',root.name,'--------')

#输入节点名称，查询节点的huffman code
def searchcode(huffmantree,node):
    node_list=[item.name for item in huffmantree.nodes]
    node_id=node_list.index(node)
    node_code=huffmantree.nodes[node_id].code
    print('--------node name is:',node,'--------','huffman code is:',node_code,'--------')
    return node_code


def CreateTree():
    text,sentense_list = fenci.getcorpus()
    leaf_nodes=LeafNode(text)
    tree=Huffman()
    tree_nodes=tree.bulid_tree(leaf_nodes)
    tree.huffmancoding(tree_nodes)
    return tree

if __name__ == '__main__':
    text={'a':4,'b':10,'c':100,'d':7,'e':19}
    # text,sentense = fenci.getcorpus()
    leaf_nodes=LeafNode(text)
    tree=Huffman()
    tree_nodes=tree.bulid_tree(leaf_nodes)
    for i in tree_nodes:
        print(i.name)
    tree.huffmancoding(tree_nodes)
    searchcode(tree,'a')
    searchcode(tree,'b')
    huffmandecoding(tree,'1001')

