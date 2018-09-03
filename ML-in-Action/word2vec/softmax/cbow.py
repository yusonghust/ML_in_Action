import numpy as np
import math
import random
import tensorflow as tf
import huffman

class Cbow():
    def __init__(self,
                words=None, #单词列表
                corpus=None, #训练使用的语料库
                huffmantree=None #根据节点创建的huffman树
                ):
        self.learning_rate=0.001
        self.representation_size=128
        self.windows=3 #单边窗口长度
        self.batchsize=300
        self.epochs=10000
        self.words = words
        self.corpus = corpus
        self.huffmantree = huffmantree
        self.embeddings = tf.Variable(tf.random_normal((len(self.words),self.representation_size)))
        self.thetas = tf.Variable(tf.random_normal((len(self.words)-1,self.representation_size)))


    def generate_batch(self):
        words_list = self.words
        batchsize = self.batchsize
        Contexts = []
        # Centers = []
        Paths = []
        windows = self.windows
        while (len(Contexts)<batchsize):
            x = random.choice(self.corpus)
            for i in range(len(x)):
                if((i+1+2*windows)>len(x)):
                    break
                start = i
                center = i+windows
                end = i+2*windows
                context =[words_list.index(word) for word in x[start:center-1]]+[words_list.index(word) for word in x[center+1,end]]
                center = words_list.index(x[center])
                center_word_code = [int(j) for j in huffman.searchcode(self.huffmantree,x[center])]
                path=[]
                root=self.huffmantree.root
                path.append(root.name)
                for j in center_word_code:
                    if(j==1):
                        path.append(root.left_child.name)
                        root=root.left_child
                    else:
                        path.append(root.right_child.name)
                        root=root.right_child


                Contexts.append(context)
                # Centers.append(center)
                Paths.append(center_word_code)
        assert(len(Contexts)==len(Paths)==batchsize)
        return Contexts,Paths






    def train(self):

        #input data 喂进来的数据是上下文节点在self.words中的Index矩阵
        contexts = tf.placeholder(tf.int32,shape=[self.batchsize,2*self.windows])
        # center_path = tf.placeholder(tf.int32,shape=[self.batchsize]) #目标节点的index列表
        # words = tf.placeholder(tf.string,shape=[len(self.words)]) #单词表


        contexts_nodes_embeddings = tf.nn.embedding_lookup(self.embeddings,contexts)

        projection_layer = tf.reduce_sum(contexts_nodes_embeddings,1)

        # center_nodes = tf.nn.embedding_lookup(words,centers)
        center_nodes_path =

