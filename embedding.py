from logger import Logger
import logging
import os
import math
import re

from parameter import path, hyper_parameter

from gensim.models import word2vec 
from gensim.models import fasttext

import numpy as np
import pandas as pd

lg = Logger()

class Text_embedding:
    def __init__(self):
        self.mode = 'fasttext'
        self.vocab = 'word_seg'
        model_path = os.listdir(os.path.join(path.MODEL_PATH, self.mode))
        
        pattern = 'dim[0-9]+_epoch[0-9].model$'
        
        isEmbed = False     # 判断是否存在模型
        for model_name in model_path: 
            if re.match(pattern, model_name):
                isEmbed = True
                break

        if not isEmbed:
            lg.info("未检测到预训练模型，开始训练词向量模型")
            self.model = self.embedding_train(self.mode, self.vocab)
        else:
            lg.info("加载词向量模型……")
            self.model = self.loads_model(self.mode)


    def embedding_train(self, mode, vocab):
        '''
        Embedding训练
        
        参数列表
        -------
        mode : 词向量模型的类型，fasttext或者word2vec，默认为fasttext
        vocab : 需要训练的语料库路径，默认为相对路径

        返回值
        -------
        : 训练得到词向量的模型
        '''
        vocab_path = os.path.join(path.TEXT_PATH, vocab)
        sentences = word2vec.PathLineSentences(vocab_path)
        lg.info(f"{mode}词向量模型开始训练,\
            词向量维度为{hyper_parameter.dimension},\
            迭代次数为{hyper_parameter.epochs}")
        # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        
        if mode == 'fasttext':
            model = fasttext.FastText(sentences, 
                vector_size=hyper_parameter.dimension, 
                window=20, epochs=hyper_parameter.epochs)
        elif mode == 'word2vec':
            model = word2vec.Word2Vec(sentences, 
                vector_size=hyper_parameter.dimension, 
                window=20, epochs=hyper_parameter.epochs)
        else:
            lg.warning("%(message)s", "未知词向量模型！")

        lg.info("模型保存中……")
        model.save(path.MODEL_PATH+f'{mode}/dim{hyper_parameter.dimension}_epoch{hyper_parameter.epochs}.model')
        lg.info("保存模型完毕")
        
        return model


    def get_embedding(self, model, vocab, mode, save=True):
        '''
        获取指定列表下所有人的文本表示
        
        参数列表
        -------
        model : 词向量模型对象
        vocab : 文本列表
        mode : 词向量模型的类型
        save : 是否保存，默认为否

        返回值
        -------
        doc_vec : ndarry类型的时间线词向量表示
        '''
        lg.info("-----------获取词向量-----------")
        vocab_path = os.path.join(path.TEXT_PATH, vocab)
        files = os.listdir(vocab_path)
        embedding =  pd.DataFrame()
        count = 0
        for file in files:
            count += 1
            user_id = file.split('.')[0]
            embedding = pd.concat([embedding, self.get_word_vector(user_id, model, mode)])
            lg.info(f"已获得用户{user_id}推文向量表示, 进度{count}/{len(files)}")
        
        if save == True:
            if not os.path.exists(path.EMBED_PATH+mode):
                os.mkdir(path.EMBED_PATH+mode)
            embed_path = os.path.join(path.EMBED_PATH, mode, 'fasttext_embedding.csv')
            embedding.to_csv(embed_path, encoding='utf-8', mode='w+')
        return embedding


    def get_word_vector(self, user_id, model=None, mode="fasttext"):
        '''
        获取一个用户的时间线词向量表示
        
        参数列表
        -------
        user_id : 用户id
        model : 词向量模型对象，默认为None
        mode : 词向量模型的类型，fasttext或者word2vec，默认为fasttext

        返回值
        -------
        doc_vec : ndarry类型的时间线词向量表示
        '''
        if model == None:
            lg.info("无传入模型，加载预定模型")
            model = self.loads_model(mode)

        tweets = []
        filepath = os.path.join(path.TEXT_PATH, self.vocab, user_id+'.txt')
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                tweets.append(line.strip().split(' '))

        timeline_vec = np.zeros((1, hyper_parameter.dimension))
        start = 0
        stint = 0
        day_mean = 32
        
        while stint < math.ceil(len(tweets)/float(day_mean)):
            # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
            if start + day_mean >= len(tweets):
                sentences = tweets[start:-1]
            else:
                sentences = tweets[start: start+day_mean]

            stint_matrix = np.zeros((len(sentences), hyper_parameter.dimension), dtype=float)
            for i in range(len(sentences)):
                if len(sentences[i][0]) != 0:
                    sen_matrix = np.zeros((len(sentences[i]), hyper_parameter.dimension), dtype=np.float) # 矩阵的每一行为该句每一个词向量
                    for j in range(len(sentences[i])):
                        sen_matrix[j] = model.wv[sentences[i][j]]
                    sen_vec = np.dot(np.ones((1, len(sentences[i])), dtype=np.float), sen_matrix)
                    norm = np.linalg.norm(sen_vec, axis=1, keepdims=True)
                    sen_vec = sen_vec / norm
                    stint_matrix[i] = sen_vec
                else:
                    continue
            
            if stint_matrix.shape[0] != 0:
                stint_vec = np.sum(stint_matrix, axis=0)/stint_matrix.shape[0]  # 平均文本表示
                timeline_vec += stint_vec * math.exp((-1)*stint)  # 加权平均文本表示
            
            stint += 1
            start += day_mean

        id_df = pd.DataFrame([user_id], columns=['NO'], dtype='str')
        tl_df = pd.DataFrame(timeline_vec)

        return pd.concat([id_df, tl_df], axis=1)
            

    def loads_model(self, mode='fasttext'):
        model_name = f'dim{hyper_parameter.dimension}_epoch{hyper_parameter.epochs}.model'
        model_path = os.path.join(path.MODEL_PATH, mode, model_name)
        
        if mode == 'fasttext':
            model = fasttext.FastText.load(model_path)
        elif mode == 'word2vec':
            model = word2vec.Word2Vec.load(model_path)
        
        return model


    # def loads_ft_model(model_name):
    #     model_path = os.path.join(path.MODEL_PATH, 'fasttext/'+model_name)
    #     model = fasttext.FastText.load(model_path)
    #     return model


    # def get_word_vector_ft(path):
    #     sentences = []
    #     row_num = 0
    #     with open(path, 'r', encoding='utf-8') as file:
    #         # lg.info(f"文件{path}内容获取")
    #         for line in file:
    #             row_num += 1
    #             sentences.append(line.strip().split(' '))
        
    #     stint = row_num / 30
    #     line30_time = 0

    #     model = loads_ft_model('fasttext/dim300_epoch5.model')

    #     tl_ave = np.zeros(hyper_parameter.dimension)
    #     sen_ave = np.zeros(hyper_parameter.dimension) #训练词向量时的超参，需要一致，下同
    #     inter_ave = np.zeros(hyper_parameter.dimension) #发帖间隔内的词向量

    #     line_num = 0 #间隔计数器
    #     sum_line_num = 0 #总行数
    #     for words in sentences:
    #         sum_line_num += 1
    #         line_num += 1
    #         count = 0
    #         sen_ave[:] = 0
    #         for word in words:
    #             if(word in model.wv.index_to_key):
    #                 vec_word = model.wv[word]
    #                 count += 1
    #                 sen_ave += vec_word
    #         if count != 0:
    #             sen_ave = sen_ave/count
    #         inter_ave += sen_ave

    #         if line_num == 30:
    #             line30_time += 1
    #             inter_ave = inter_ave/line_num
    #             line_num = 0
    #             k = 1-((line30_time-1)/(stint+1))  #线性递减系数
    #             tl_ave += k * inter_ave
    #             inter_ave[:] = 0
    #     if line_num < 30 and line_num != 0:
    #         tl_ave += 1/(stint+1) * (inter_ave/line_num)

    #     tl_ave = tl_ave/stint
    #     return tl_ave


