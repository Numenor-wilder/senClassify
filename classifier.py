import logging
import os

from embedding import Text_embedding
from parameter import path
from parameter import hyper_parameter as hparam

from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score
from sklearn.model_selection import train_test_split
from joblib import dump,load
import numpy as np
import pandas as pd


LABEL_PATH = r'.\data\labeled_data'

# lg = Logger()


class Classifier(object):
    def __init__(self, etype='fasttext') -> None:
        self.mode= etype


    def svm_training(self, features, labels):
        logging.info("SVM分类器训练开始")
        if np.any(np.isnan(features)) or np.any(np.isnan(labels)):
            logging.warning("训练数据有空值！！！")
            return

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

        clf = SVC(C=1,kernel='rbf',gamma=0.1)
        while i < 10:     
            for col in labels.columns:
                logging.info(f"第{i+1}次训练")
                x_train, x_test, y_train, y_test = train_test_split(features, labels[col], test_size=0.2)

                clf.fit(x_train, y_train)
                y_pred = pd.Series(clf.predict(x_test))

                acc[col] += round(clf.score(x_test, y_test),4)
                # recall[col] += recall_score(y_test, y_pred, pos_label='positive',average='weighted')
                prec[col] += precision_score(y_test, y_pred, pos_label='positive', average='weighted')
                f1[col] += f1_score(y_test, y_pred, pos_label='positive', average='weighted')
                logging.info(f"{col}测试集准确率: {round(clf.score(x_test, y_test),4)}")
                # logging.info(f"{col}测试集召回率: {recall_score(y_test, y_pred, pos_label='positive', average='weighted')}")
                logging.info(f"{col}测试集precision: {round(precision_score(y_test, y_pred, pos_label='positive', average='weighted'),4)}")
                logging.info(f"{col}测试集F1值: {round(f1_score(y_test, y_pred, pos_label='positive', average='weighted'),4)}")

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


    # def get_training_data(self, label_path):
    #     if self.epath:
    #         features, labels = self.load_training_data(self.epath, label_path, self.mode)
    #     else:
    #         lg.warning("缺少文本特征表示！！！")
    #         features = None 

    #     return features, labels



    def load_training_data(self, emb_filepath, label_filepath, mode=""):
        logging.info("载入特征数据和标签数据...")
        if not mode:
            mode = self.mode
        
        emb_path = os.path.join(path.EMBED_PATH, mode, emb_filepath)
        label_path = os.path.join(LABEL_PATH, label_filepath)

        if os.path.exists(emb_path):
            features = pd.read_csv(emb_path, dtype={'NO': 'str'}).sort_values('NO')
        else:
            logging.warning("文本特征数据路径不存在！！！")
            features = None

        if os.path.exists(label_path):
            labels = pd.read_csv(label_path, dtype={'NO': 'str'}).sort_values('NO')
        else:
            logging.warning("标签文件路径不存在！！！")
            labels = None
            

        # 当数据不一致的时候，选择其中相同的部分（临时处理）
        if features is not None and labels is not None:
            merge = features.merge(labels, how='inner', on='NO')
            features_ = features.loc[features['NO'].isin(merge['NO'])]
            labels_ = labels.loc[labels['NO'].isin(merge['NO'])]

            return features_.iloc[:,1:], labels_.iloc[:,1:]
        else:
            return features, labels
                


    # 生成训练数据
    def generate_training_data(self, doc_path="", label_filepath=""):
        logging.info("生成训练数据...")
        # params = hparam(300,5,20)
        if not doc_path:
            logging.warning("缺少文本特征输入！！！")
            return
        if not label_filepath:
            logging.warning("缺少标签数据！！！")
            return

        text_embedding = Text_embedding()
        if text_embedding.model == None:
            text_embedding.model = text_embedding.embedding_train(vocab=doc_path)
        embed = text_embedding.get_embedding(doc=doc_path,save=False)
        
        label_path = os.path.join(LABEL_PATH, label_filepath)
        labels = pd.read_csv(label_path, dtype={'NO': 'str'}).sort_values('NO')
        features = embed

        if features.shape[0] == labels.shape[0]:  # 临时处理
            return features.iloc[:,1:], labels.iloc[:,1:]

        else: # 当数据不一致的时候，选择其中相同的部分
            merge = features.merge(labels, how='inner', on='NO')
            features_ = features.loc[features['NO'].isin(merge['NO'])]
            labels_ = labels.loc[labels['NO'].isin(merge['NO'])]

            return features_.iloc[:,1:], labels_.iloc[:,1:]


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