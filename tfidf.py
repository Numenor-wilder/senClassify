import logging
import os
# import glob

from parameter import path

from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import numpy as np


class Tfidf():
    dataset = []
    dct = None
    corpus = None
    model = None
    def __init__(self, doc): # doc: 构建语料库文本的文件夹路径
        self.doc=doc
        self.dataset = self.make_dataset(self.doc)
        self.dct = self.get_dict(self.doc)
        self.corpus = self.get_corpus(self.doc)
        self.model = self.get_tfidf_model(self.doc)


    def get_tfidf_model(self, model_path):
        '''
        从给定路径加载TF-IDF模型
        
        参数
        ----
        model_path : 模型路径
        '''
        model_name = os.path.split(model_path)[-1]
        if not os.path.exists(path.TFIDF_MODEL_PATH):
            logging.warning("模型未训练！！！")
            os.mkdir(path.TFIDF_MODEL_PATH)
            return
        else:
            model_path = os.path.join(path.TFIDF_MODEL_PATH, model_name+'.model')
            logging.info("加载TF-IDF模型...")
            if os.path.exists(model_path):
                return TfidfModel.load(model_path)
            else:
                logging.warning("未找到指定的模型！！！")
                return


    def train_tfidf_model(self, corpus_path):
        '''
        根据给定语料库训练TF-IDF模型，若语料库不存在则根据数据集和字典生成语料库
        将训练好的TF-IDF模型保存到指定路径
        
        参数
        ----
        corpus_path : 语料库路径
        '''
        corpus_name = os.path.split(corpus_path)[-1]

        self.corpus = self.get_corpus(corpus_name)

        if self.corpus is None:
            self.corpus = self.make_corpus(self.dataset, self.dct)
        else:
            self.corpus = self.corpus.tolist()
        self.model = TfidfModel(corpus=self.corpus)  # fit model

        model_name = os.path.split(corpus_path)[-1]
        if not os.path.exists(path.TFIDF_MODEL_PATH):
            os.mkdir(path.TFIDF_MODEL_PATH)
        model_path = os.path.join(path.TFIDF_MODEL_PATH, model_name+'.model')
        self.model.save(model_path)

        return self.model

        
    def get_corpus(self, corpus_path):
        '''
        从指定路径加载语料库
        
        参数
        ----
        corpus_path : 语料库路径
        '''
        corpus_name = os.path.split(corpus_path)[-1]
        if ".npy" not in corpus_name:
            corpus_name += ".npy"
        corpus_path = os.path.join(path.CORPUS_PATH, corpus_name)

        if os.path.exists(corpus_path):
            logging.info("加载语料库...")
            corpus = np.load(corpus_path, allow_pickle=True)
        else:
            logging.warning("语料库不存在！！！")
            return
        return corpus

    
    def make_corpus(self, dataset, dct):
        '''
        通过字典和数据集生成语料库
        
        参数
        ----
        dataset : 数据集, list类型
        dct : 字典（termid-term表示）
        '''
        corpus = [dct.doc2bow(line) for line in dataset]  # convert corpus to BoW format
        
        corpus_name = os.path.split(self.doc)[-1]
        if not os.path.exists(path.CORPUS_PATH):
            os.mkdir(path.CORPUS_PATH)
        corpus_path = os.path.join(path.CORPUS_PATH, corpus_name)

        while(1):
            if not os.path.exists(corpus_path):
                np.save(corpus_path, corpus, allow_pickle=True)
                break
            else:   # 若文件存在，则在文件后缀下划线
                corpus_path += '_'

        return corpus


    def get_dict(self, doc):
        '''
        根据指定路径获取字典
        
        参数
        ----
        doc : 字典语料路径
        '''
        dict_name = os.path.split(doc)[-1]
        dict_path = os.path.join(path.CORPUS_PATH, 'dic', dict_name)
        
        if os.path.exists(dict_path):
            logging.info("加载字典...")
            dct = Dictionary.load(dict_path)
            return dct
        else:
            logging.warning("不存在指定的字典！！！")
            return


    def make_dict(self, doc, dataset=None):
        '''
        根据数据集生成字典
        
        参数
        ----
        doc : 字典语料路径
        dataset : 数据集
        '''
        if dataset == None:
            dataset = self.make_dataset(doc)
        
        doc_name = os.path.split(doc)[-1]
        dct_path = os.path.join(path.CORPUS_PATH, 'dic', doc_name)
        dct = None

        dct = Dictionary(dataset) # fit dictionary
    
        if not os.path.exists(path.CORPUS_PATH):
            os.mkdir(path.CORPUS_PATH)
        if not os.path.exists(os.path.join(path.CORPUS_PATH, 'dic')):
            os.mkdir(os.path.join(path.CORPUS_PATH, 'dic'))
        
        while(1):
            if not os.path.exists(dct_path):
                dct.save(dct_path)
                break
            else:   # 若文件存在，则在文件后缀下划线
                dct_path += '_'
    
        return dct


    def make_dataset(self, doc):
        '''
        生成数据集
        
        参数
        ----
        doc : 文本路径
        '''
        dataset = []
        doc_path = os.path.join(path.TEXT_PATH, doc)
        if not os.path.exists(doc_path):
            logging.warning("没有找到指定的数据集")
            return
        
        files = os.listdir(doc_path)
        for file in files:
            filepath=os.path.join(doc_path, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                vocabset = []
                for line in f:
                    if line.strip() == u'': # 去除空行
                        continue
                    vocabset.extend(line.strip().split(' '))
                dataset.append(vocabset)
        return dataset