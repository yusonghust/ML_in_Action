import jieba
from collections import Counter



jieba.add_word('诛仙剑')
jieba.add_word('青云门')
jieba.add_word('草庙村')
jieba.add_word('死灵渊')
jieba.add_word('鬼厉')
jieba.add_word('田不易')
jieba.add_word('田灵儿')
jieba.add_word('道玄')
jieba.add_word('道玄真人')
jieba.add_word('张小凡')
jieba.add_word('碧瑶')
jieba.add_word('陆雪琪')



except_list=[]
f0=open('./stop_word.txt','r',errors='ignore')
for line in f0:
    ls=line.strip().split(' ')
    except_list.append(ls[0])
f0.close()


def cut_word(path):
    raw_word_list=[]
    sentense_list=[]
    print('read txt')
    f=open(path,'r',errors='ignore',encoding='gbk')
    for line in f:
        ls=line.strip()
        if(len(ls)>0):
            seg_list=jieba.cut(ls,cut_all=False)
            wd=[i for i in seg_list if i not in except_list]
            raw_word_list=raw_word_list+wd
            ls=ls.split('。')
            for i in ls:
                sentense = [k for k in jieba.cut(i,cut_all=False) if k not in except_list]
                if(len(sentense)>6):
                    sentense_list.append(sentense)
    word_count=dict(Counter(raw_word_list).most_common(30000))
    f.close()
    return word_count,sentense_list

def getcorpus():
    # sets()
    path='./zhuxian.txt'
    word_count,sentense_list = cut_word(path)
    return word_count,sentense_list
