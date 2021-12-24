import sys
sys.path.append("/home/work/code")

import os
import logging

from common.parameters import GlobalPath
from common.parameters import EmbeddingHyperParma as EmbHparam
from common.utility import path_exist_operate
import numpy as np

from gensim.models import word2vec 
from gensim.models import fasttext
from bert_serving.client import BertClient


class Text_embedding:
    def __init__(self, vocab, mode=""):
        
        '''
        参数列表
        -------
        vocab : 语料库名称
        mode : 词向量模型的类型，默认为fasttext
        '''
        self.vocab = vocab        # 训练词向量的语料库
        self.params = EmbHparam(512, 5, 10)   # 超参
        self.model = None        # 词向量模型
        self.mode = ""           # 模型类型        

        
        if mode == "":
            self.mode = "fasttext"
        else:
            self.mode = mode
        
        # self.emb_file = emb_file
        

        model_path = os.path.join(GlobalPath.EMBED_MODEL_PATH, self.mode, self.vocab+'.model')
        self.model, err = self.get_model(model_path)
        
        if err:
            path_exist_operate(model_path)   # 处理文件路径问题
            logging.info("未发现对应语料的词向量模型")


   

    def embedding_model_train(self, vocab_name="", params=None, mode="", implicit=False):
        '''
        Embedding训练
        
        参数列表
        -------
        :param vocab_name: 需要训练的语料库文件名
        :param params: 需指定的超参
        :param mode: 词向量模型的类型，默认为fasttext
        :param implicit: 隐式特征开启标识符, 默认为False
        返回值
        -------
        : 训练得到词向量的模型
        '''
        if params == None:
            params = self.params

        if mode == "":  # 默认采用实例成员模型训练类型
            mode = self.mode

        vocab_pre_path = GlobalPath.CLEAN_DATA_NORMAL
        if implicit:    # 隐式特征
            vocab_name += '_implicit'
        vocab_path = os.path.join(vocab_pre_path, vocab_name)

        if vocab_name and os.path.exists(vocab_path):
            sentences = word2vec.PathLineSentences(vocab_path)
            logging.info(f"{mode}词向量模型开始训练,词向量维度为{params.dimension},迭代次数为{params.epochs},窗口大小为{params.window}")
            
            fasttext_model = None
            if mode == 'fasttext':
                fasttext_model = fasttext.FastText(sentences, 
                    vector_size=params.dimension, 
                    window=params.window, epochs=params.epochs)
            else:
                logging.warning("暂不支持该词向量模型!!!")
                return
        
            model_save_path = os.path.join(GlobalPath.EMBED_MODEL_PATH, mode, f'{vocab_name}.model')
            path_exist_operate(model_save_path)

            while(1):
                if not os.path.exists(model_save_path):
                    logging.info("模型保存中……")
                    fasttext_model.save(model_save_path)
                    logging.info("保存模型完毕")
                    break
                else:
                    os.remove(model_save_path)
        
            self.model = fasttext_model
            return fasttext_model
        else:
            path_exist_operate(vocab_path)
            raise FileNotFoundError



    def get_embedding_single(self, doc_path, model):
        '''
        获取单个文件的文本表示
        
        参数列表
        -------
        doc_path : 文档路径
        model : 词向量模型对象

        返回值
        -------
        embedding_norm : ndarry类型的时间线词向量表示
        '''

        count = 0
        user_embedding = np.ndarray((model.vector_size,))
        with open(doc_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip() == u'': # 去除空行
                    continue
                words = line.strip().split(' ')
                count += len(words)
                for w in words:
                    user_embedding += model.wv[w]  
            user_embedding /= count
                     
        return user_embedding


    def get_model(self, model_path):
        '''
        载入词向量模型

        :param model_path: 模型路径
        '''
        model = None

        try:
            model = fasttext.FastText.load(model_path)
        except FileNotFoundError as err:
            logging.warning("模型不存在, %s", err.args[1])
            return model, err
        else:
            return model, None
    
