import os 

from logger import Logger
from embedding import Text_embedding
from parameter import hyper_parameter, path

from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


LABEL_PATH = './data/labeled_data/'
TRAIN_PATH = './data/training_data/'
EMBED_PATH = './embedding/'

lg = Logger()


class Classifier(object):
    def __init__(self, etype='fasttext', epath='') -> None:
        self.mode= etype
        self.epath = epath

    def svm_training(self, features, labels):
        lg.info("SVM分类器训练开始")
        if np.any(np.isnan(features)) or np.any(np.isnan(labels)):
            lg.warning("训练数据有空值！！！")

        i = 0
        acc = {}
        recall = {}
        f1 = {}
        prec = {}

        for col in labels.columns:
            acc[col] = 0
            recall[col] = 0
            f1[col] = 0
            prec[col] = 0
        while i < 10:     
            for col in labels.columns:
                lg.info(f"第{i+1}次训练")
                x_train, x_test, y_train, y_test = train_test_split(features, labels[col], test_size=0.2)
                clf = SVC(C=1,kernel='rbf',gamma=0.1)
                clf.fit(x_train, y_train)
                y_pred = pd.Series(clf.predict(x_test))

                acc[col] += round(clf.score(x_test, y_test),4)
                # recall[col] += recall_score(y_test, y_pred, pos_label='positive',average='weighted')
                f1[col] += f1_score(y_test, y_pred, pos_label='positive', average='weighted')
                prec[col] += precision_score(y_test, y_pred, pos_label='positive', average='weighted')
                lg.info(f"{col}测试集准确率: {round(clf.score(x_test, y_test),4)}")
                # lg.info(f"{col}测试集召回率: {recall_score(y_test, y_pred, pos_label='positive', average='weighted')}")
                lg.info(f"{col}测试集F1值: {round(f1_score(y_test, y_pred, pos_label='positive', average='weighted'),4)}")
                lg.info(f"{col}测试集precision: {round(precision_score(y_test, y_pred, pos_label='positive', average='weighted'),4)}")

            i += 1

        for col in labels.columns:
            acc[col] /= 10
            recall[col] /= 10
            f1[col] /= 10
            prec[col] /= 10
            print(f"{col} mean accuracy:", round(acc[col],4))
            # print(f"{col} mean recall:", round(recall[col],4))
            print(f"{col} mean f1:", round(f1[col],4))
            print(f"{col} mean precision:", round(prec[col],4))





    # 获得训练数据
    def get_training_data(self, filepath=''):
        if filepath != '':
            lg.info("载入自定义词向量文件")
            embed_path = os.path.join(path.EMBED_PATH, self.mode, filepath)
            embed = pd.read_csv(embed_path).astype({'NO': 'str'}).iloc[:,1:]
        else:
            lg.info("载入模型词向量")
            text_embed = Text_embedding()
            embed = text_embed.get_embedding(text_embed.model, text_embed.vocab, text_embed.mode)
        
        label_path = os.path.join(LABEL_PATH, 'all_features.csv')
        label = pd.read_csv(label_path, dtype={'NO': 'str'}).sort_values('NO').iloc[:,1:]

        lg.info("生成训练数据")
        train = pd.merge(embed, label)
        train.to_csv(TRAIN_PATH+'training.csv',encoding='utf-8')

        return train.iloc[:, 1:hyper_parameter.dimension+1], train.iloc[:, hyper_parameter.dimension+1:]



    # #生成训练数据
    # def generate_training_data(label_path, embeding=None, embed_path='', mode='ft'):
    #     if embed_path != '':
    #         embed = pd.read_csv(embed_path).rename(columns = {'Unnamed: 0': 'NO'}).astype({'NO': 'str'})
    #     if embeding:
    #         embed = embeding
        
    #     label = pd.read_csv(label_path, dtype={'NO': 'str'}).sort_values('NO').iloc[:,1:]

    #     train = pd.merge(embed, label)
    #     train.to_csv(TRAIN_PATH+'training.csv',encoding='utf-8')

    #     return train.iloc[:, 1:hyper_parameter.dimension+1], train.iloc[:, hyper_parameter.dimension+1:]


    # 打印数据标签值分布
    # label_file = os.path.join(LABEL_PATH,'all_features.csv')
    # triplets = count_share(label_file)
    # for tri in triplets[1:]:
    #     print(f"{tri.name}的标签分布: ", end='')
    #     i = 0
    #     for item in tri:
    #         print(f"{tri.index[i]}:{item} ", end="")
    #         if i < len(tri):
    #             i += 1
    #     print()