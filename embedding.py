
import glob
import os
import math
import logging

from parameter import path
from parameter import hyper_parameter as hparam

import numpy as np
import pandas as pd

from gensim.models import word2vec 
from gensim.models import fasttext
from bert_serving.client import BertClient
# from gensim.models import doc2vec



class Text_embedding:
    vocab = ""          # 训练词向量的语料库
    params = hparam(300, 5, 20)   # 超参
    model = None        # 词向量模型
    mode = ""           # 模型类型

    def __init__(self, params=None, vocab="", mode=""):
        '''
        参数列表
        -------
        params : 模型可调超参
        vocab : 需要训练的语料库路径
        mode : 词向量模型的类型，默认为fasttext
        '''
        if params != None:
            self.params = params
        if vocab != "":
            self.vocab = vocab
        
        if mode == "":
            self.mode = "fasttext"
        else:
            self.mode = mode
        
        # self.emb_file = emb_file
        
        if self.mode == "fasttext":
            model_path = os.path.join(path.EMBED_MODEL_PATH, self.mode)
            if os.path.exists(model_path):
                if len(os.listdir(model_path)): # 找到最近修改时间的模型并加载
                    e_models_list = glob.glob(os.path.join(model_path, '*'))
                    latest_model = max(e_models_list, key=os.path.getmtime)
                    self.model = fasttext.FastText.load(latest_model)
                else:
                    pass
            else:
                os.mkdir(model_path)
        elif self.mode == "bert":
            self.model = BertClient()


    def embedding_train(self, vocab="", params=None, mode=""):
        '''
        Embedding训练
        
        参数列表
        -------
        vocab : 需要训练的语料库路径，默认为相对路径
        params : 需指定的超参
        mode : 词向量模型的类型，默认为fasttext

        返回值
        -------
        : 训练得到词向量的模型
        '''
        if params == None:
                params = self.params

        if vocab == "":
            if self.vocab:
                vocab = self.vocab
            else:
                logging.warning("缺少训练语料库！！！")
                return

        if mode == "":  # 默认采用实例成员模型训练类型
            mode = self.mode

        vocab_name = os.path.split(vocab)[-1] #
        vocab_path = os.path.join(path.TEXT_PATH, vocab)
        sentences = word2vec.PathLineSentences(vocab_path)
        logging.info(f"{mode}词向量模型开始训练,词向量维度为{params.dimension},迭代次数为{params.epochs},窗口大小为{params.window}")
        # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        
        if mode == 'fasttext':
            model = fasttext.FastText(sentences, 
                vector_size=params.dimension, 
                window=params.window, epochs=params.epochs)
        else:
            logging.warning("暂不支持该词向量模型！！！")
            return
        
        mode_path = os.path.join(path.EMBED_MODEL_PATH, mode)
        if not os.path.exists(mode_path):
            os.mkdir(mode_path)
        
        model_save_path = os.path.join(mode_path, f'{vocab_name}.model')
        logging.info("模型保存中……")
        model.save(model_save_path)
        logging.info("保存模型完毕")
        
        self.model = model

        return model


    def get_embedding(self, model=None, doc="", mode="", tfidf=None, save=True):
        '''
        获取指定列表下所有人的文本表示
        
        参数列表
        -------
        model : 词向量模型对象
        doc : 文档列表
        tfidf : TF-IDF模型
        mode : 词向量模型的类型
        save : 是否保存，默认为否

        返回值
        -------
        embedding_norm : ndarry类型的时间线词向量表示
        '''

        if model == None :
            if self.model:
                model = self.model
            else:
                logging.warning("没有载入预训练的词向量模型！！！")
                return
        
        if doc == "":
            if self.vocab:
                doc = self.vocab
            else:
                logging.warning("缺少需要被表示的文本文件！！！")
                return

        if mode == "":
            mode = self.mode

        doc_path = os.path.join(path.TEXT_PATH, doc)
        files = os.listdir(doc_path)
        embedding = pd.DataFrame()
        count = 0

        for index in range(len(files)):
            user_vector = pd.DataFrame(self.get_ft_tfidf_vector(self.model, 
                                            tfidf.model, tfidf.corpus, tfidf.dct,
                                            index))
            embedding = embedding.append(user_vector.T, ignore_index=True)
            logging.info(f"已获取{files[index]}推文向量表示, 进度{index+1}/{len(files)}")
                    

        embedding_norm = (embedding - embedding.mean()) / (embedding.max() - embedding.min())
        
        if save == True:    # 保存embedding文件
            if not os.path.exists(os.path.join(path.EMBED_PATH, mode)):
                os.mkdir(os.path.join(path.EMBED_PATH, mode))

            doc_name = os.path.split(doc)[-1]
            embed_path = os.path.join(path.EMBED_PATH, mode, doc_name)

            user_id = pd.DataFrame([file.split('.')[0] for file in files], columns=['NO'])
            user_embedding = pd.concat([user_id, embedding_norm], axis=1)

            while(1):
                if not os.path.exists(embed_path):
                    user_embedding.to_csv(embed_path, encoding='utf-8', mode='w', index=False)
                    break
                else:   # 若文件存在，则在文件后缀下划线
                    embed_path += '_'
        
        return user_embedding
    

    def get_ft_tfidf_vector(self, e_model, t_model, corpus, dct, index):
        '''
        计算文档的TF-IDF加权向量表示
        
        参数列表
        -------
        e_model : 词向量模型对象
        t_model : TF-IDF模型
        corpus : 语料库
        dct : 字典
        index : 文档索引号

        返回值
        -------
        weight_vector : 文档的TF-IDF加权向量表示
        '''
        vector = t_model[corpus[index]]
        
        weight_vector = np.zeros((e_model.vector_size,), dtype=float)

        for item in vector:
            word_vector = e_model.wv[dct[item[0]]]
            weight_vector += item[1] * word_vector

        return weight_vector



    # def get_vector(self, doc, user_id, model, mode):
        '''
        获取一个用户的时间线词向量表示
        
        参数列表
        -------
        doc : 需要被表示的文本文件
        user_id : 用户id
        model : 词向量模型对象
        mode : 词向量模型的类型

        返回值
        -------
        doc_vec : ndarry类型的时间线词向量表示
        '''
        tweets = []
        filepath = os.path.join(path.TEXT_PATH, doc, user_id)
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                tweets.append(line.strip().split(' '))

        timeline_vec = np.zeros((1, model.vector_size))
        start = 0
        stint = 0
        day_mean = 32
        
        while stint < math.ceil(len(tweets)/float(day_mean)):
            if start + day_mean >= len(tweets):
                sentences = tweets[start:-1]
            else:
                sentences = tweets[start: start+day_mean]
            
            stint_matrix = np.zeros((len(sentences), model.vector_size), dtype=float)
            if mode == 'fasttext':
                stint_matrix = self.get_ft_vector(sentences, model)
            if mode == 'bert':
                stint_matrix = self.get_bert_vector(sentences, model)

            if stint_matrix.shape[0] != 0:
                stint_vec = np.sum(stint_matrix, axis=0)/stint_matrix.shape[0]  # 平均文本表示
                timeline_vec += stint_vec * math.exp((-1)*stint)  # 加权平均文本表示
            
            stint += 1
            start += day_mean

        id_df = pd.DataFrame([user_id], columns=['NO'], dtype='str')
        tl_df = pd.DataFrame(timeline_vec)

        return pd.concat([id_df, tl_df], axis=1)
            

    # def get_ft_vector(self, sentences, model):
        stint_matrix = np.zeros((len(sentences), model.vector_size), dtype=float)
        for i in range(len(sentences)):
            if len(sentences[i][0]) != 0:
                sen_matrix = np.zeros((len(sentences[i]), model.vector_size), dtype=np.float) # 矩阵的每一行为该句每一个词向量
                for j in range(len(sentences[i])):
                    sen_matrix[j] = model.wv[sentences[i][j]]
                sen_vec = np.dot(np.ones((1, len(sentences[i])), dtype=np.float), sen_matrix)
                norm = np.linalg.norm(sen_vec, axis=1, keepdims=True)
                sen_vec = sen_vec / norm
                stint_matrix[i] = sen_vec
            else:
                continue

        return stint_matrix


    

    def get_bert_vector(self, sentences, model):
        logging.info("bert encoding....")
        sent = []
        for item in sentences:
            sent.append(item[0])
        return model.encode(sent)


    

