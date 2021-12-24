import logging
import os
from typing import Tuple

from common.parameters import GlobalPath
from common.utility import path_exist_operate
from fine_grained.representation.embedding import Text_embedding

from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import numpy as np
import pandas as pd


class Tfidf_Embedding(Text_embedding):

    def __init__(self, vocab): # doc_path: 构建语料库文本的文件夹路径
        super(Tfidf_Embedding, self).__init__(vocab)
        self.vocab=vocab  # 需要处理文本预料路径
        self.tfidf_model, err_model = self.__get_tfidf_model(self.vocab)
        if err_model:
            self.corpus, err_corpus = self.__get_corpus(self.vocab) # 语料库
            if err_corpus:
                self.dataset = self.__make_dataset(self.vocab)  # 生成数据集
                self.dct, err_dict = self.__get_dict(self.vocab)  # 字典
                if err_dict:
                    self.dct = self.__make_dict(self.dataset)
                self.corpus = self.__make_corpus(self.dataset, self.dct)

            self.tfidf_model = self.__train_tfidf_model(self.corpus)


    def __get_tfidf_model(self, model_name):
        '''
        从给定路径加载TF-IDF模型
        
        参数
        :param model_name: 模型名称, 同数据集文件名
        ----
        返回
        :return tfidf_model, err
        '''
        model_path = os.path.join(GlobalPath.TFIDF_MODEL_PATH, model_name+'.model')
        try:
            logging.info("加载TF-IDF模型...")
            tfidf_model = TfidfModel.load(model_path)
        except FileNotFoundError as err:
            path_exist_operate(model_path)
            logging.warning("%s TF-IDF模型不存在", err)
            return None, err
        else:
            return tfidf_model, None
        


    def __train_tfidf_model(self, corpus: list):
        '''
        根据给定语料库训练TF-IDF模型
        将训练好的TF-IDF模型保存到指定路径并返回
        
        参数
        :param corpus: 语料库
        ----
        返回值
        :return tfidf_model
        '''
        logging.info("TF-IDF模型训练...") 
        self.tfidf_model = TfidfModel(corpus=corpus)  # fit model

        model_save_path = os.path.join(GlobalPath.TFIDF_MODEL_PATH, self.vocab+'.model')
        path_exist_operate(model_save_path)
        self.tfidf_model.save(model_save_path)

        return self.tfidf_model

        
    def __get_corpus(self, corpus_name: str):
        '''
        加载指定语料库并返回
        
        参数
        :param corpus_name: 语料库名称
        ----
        返回值
        :return corpus, err
        '''
        if ".npy" not in corpus_name:
            corpus_name += ".npy"
        corpus_path = os.path.join(GlobalPath.TFIDF_CORPUS_PATH, corpus_name)

        try:
            logging.info("加载语料库...")
            corpus = np.load(corpus_path, allow_pickle=True).tolist()
        except FileNotFoundError as err:
            path_exist_operate(corpus_path)
            logging.warning("%s 构造TF-IDF模型的语料库不存在", err)
            return None, err
        else:
            return corpus, None

    
    def __make_corpus(self, dataset: list, dct: Dictionary):
        '''
        通过字典和数据集生成语料库
        
        参数
        :param dataset : 数据集, list类型
        :param dct : 字典(termid-term表示)
        ----
        返回值
        :return corpus: 语料库
        '''
        logging.info("构造语料库...")
        corpus = [dct.doc2bow(line) for line in dataset]  # convert corpus to BoW format
        
        corpus_save_path = os.path.join(GlobalPath.TFIDF_CORPUS_PATH, self.vocab)
        path_exist_operate(corpus_save_path)

        while(1):
            if not os.path.exists(corpus_save_path):
                np.save(corpus_save_path, corpus, allow_pickle=True)
                break
            else: 
                os.remove(corpus_save_path)

        return corpus


    def __get_dict(self, dict_name: str):
        '''
        根据指定路径获取字典
        
        参数
        :param dict_name: 字典名
        ----
        返回值
        :return dict, err
        '''
        dict_path = os.path.join(GlobalPath.TFIDF_CORPUS_PATH, 'dict', dict_name)
        
        try:
            logging.info("加载字典...")
            dct = Dictionary.load(dict_path)
        except FileNotFoundError as err:
            path_exist_operate(dict_path)
            logging.warning("%s 生成构造语料库的字典不存在", err)
            return None, err
        else:
            return dct, None



    def __make_dict(self, dataset):
        '''
        根据数据集生成字典
        
        参数
        :param dataset: 数据集
        ----
        返回
        :return dct: 字典
        '''

        dict_save_path = os.path.join(GlobalPath.TFIDF_CORPUS_PATH, 'dict', self.vocab)
        path_exist_operate(dict_save_path)

        logging.info("构造id-term字典...")
        dct = Dictionary(dataset) # fit dictionary
        
        while(1):
            if not os.path.exists(dict_save_path):
                dct.save(dict_save_path)
                break
            else:  
                os.remove(dict_save_path)
    
        return dct


    def __make_dataset(self, vocab_name):
        '''
        生成数据集
        
        参数
        :param vocab_name: 文本路径
        ----
        返回
        :return dataset:
        '''
        logging.info("构造所需数据集...")
        dataset = []
        vocab_path = os.path.join(GlobalPath.CLEAN_DATA_NORMAL, vocab_name)

        # if parents_dir == doc_path:
        files = os.listdir(vocab_path)
        for file in files:
            file_path=os.path.join(vocab_path, file)
            try:
                f = open(file_path, 'r', encoding='utf-8')
            except FileNotFoundError as err:
                logging.error(err)
            else:
                vocabset = []
                for line in f:
                    if line.strip() == u'': # 去除空行
                        continue
                    vocabset.extend(line.strip().split(' '))
                dataset.append(vocabset)
                f.close()

        return dataset

    
    def get_tfidf_embedding(self, vocab: str, mode="fasttext", save=True):
        '''
        获取指定列表下所有人的文本表示
        
        参数列表
        -------
        :param doc: 文档列表
        :param mode: 词向量模型的类型
        :param save: 是否保存，默认为True

        返回值
        -------
        user_embedding : ndarry类型的时间线词向量表示
        '''

        vocab_path = os.path.join(GlobalPath.CLEAN_DATA_NORMAL, vocab)
        files = os.listdir(vocab_path)
        embedding = pd.DataFrame()

        for index in range(len(files)):
            # user_vector = self.get_vector(files[index], self.model, self.mode)
            user_vector = pd.DataFrame(self.__get_tfidf_vector(self.corpus, self.dct,
                                            index))
            embedding = embedding.append(user_vector.T, ignore_index=True)
            logging.info(f"已获取{files[index]}推文向量表示, 进度{index+1}/{len(files)}")
                    
        # 正则化
        embedding_norm = (embedding - embedding.mean()) / (embedding.max() - embedding.min())
        
        user_id = pd.DataFrame([file.split('.')[0] for file in files], columns=['NO'])
        user_embedding = pd.concat([user_id, embedding_norm], axis=1)

        if save == True:    # 保存embedding文件
            embed_path = os.path.join(GlobalPath.EMBED_PATH, mode, vocab)
            path_exist_operate(embed_path)
            
            suffix_info = ['d'+str(self.model.vector_size), 
                            'e'+str(self.model.epochs), 
                            'w'+str(self.model.window)]
            embed_path += '_'
            embed_path += '_'.join(suffix_info)

            while(1):
                if not os.path.exists(embed_path):
                    user_embedding.to_csv(embed_path, encoding='utf-8', mode='w', index=False)
                    break
                else: 
                    os.remove(embed_path)
        
        return user_embedding



    def __get_tfidf_vector(self, corpus, dct, index):
        '''
        计算文档的TF-IDF加权向量表示
        
        参数列表
        -------
        corpus : 语料库
        dct : 字典
        index : 文档索引号

        返回值
        -------
        weight_vector : 文档的TF-IDF加权向量表示
        '''
        vector = self.tfidf_model[corpus[index]]
        
        if self.model is None:
            logging.info("词向量模型训练")
            self.model = self.embedding_model_train(self.vocab)

        weight_vector = np.zeros((self.model.vector_size,), dtype=float)

        for item in vector:
            word_vector = self.model.wv[dct[item[0]]]
            weight_vector += item[1] * word_vector

        return weight_vector