

import logging
import os

from common.parameters import GlobalPath

from sklearn.metrics import f1_score, precision_score,accuracy_score

from joblib import load
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


class Classifier(object):
    def __init__(self, etype='fasttext') -> None:
        self.e_mode= etype
 

    def load_training_data(self, emb_filepath, label_filepath, e_mode=""):
        '''
        载入训练数据

        :param emb_filepath 文本表示文件路径
        :param label_filepath 标注数据文件路径
        :param mode 文本表示模型所采用的类型，路径的一部分

        :return features, labels 训练数据所需特征数据,标签数据, Dataframe类型
        '''

        logging.info("载入特征数据和标签数据...")
        features = None
        labels = None

        e_mode = self.e_mode
        
        try:
            if e_mode != 'fasttext':
                raise Exception(f"{e_mode}为暂不支持的词向量模型")
        except Exception as err:
            logging.error("model type is not supported for now:", err)
        
        else:
            emb_path = os.path.join(GlobalPath.EMBED_PATH, e_mode, emb_filepath)
            label_path = os.path.join(GlobalPath.LABEL_PATH, label_filepath)

            try:
                features = pd.read_csv(emb_path, dtype={'NO': 'str'}).sort_values('NO')
                labels = pd.read_csv(label_path, dtype={'NO': 'str'}).sort_values('NO')
            except FileNotFoundError as err:
                logging.error("文本特征数据路径不存在", err)
        
        finally:
            # 当数据不一致的时候，选择其中相同的部分
            if features is not None and labels is not None:
                # merge = features.merge(labels, how='inner', on='NO')
                # features_ = features.loc[features['NO'].isin(merge['NO'])].reset_index(drop=True)
                # labels_ = labels.loc[labels['NO'].isin(merge['NO'])].reset_index(drop=True)

                features_ = features.reset_index(drop=True)
                labels_ = labels.reset_index(drop=True)

                return features_.iloc[:,1:], labels_.iloc[:,1:], features_.iloc[:, 0:1]
            else:
                return features, labels


    # 重新采样smote
    def resample_smote(self, x, y):
        model_smote = SMOTE(random_state=42)
        x_smote_resamples, y_smote_resamples = model_smote.fit_resample(x, y)
        return x_smote_resamples, y_smote_resamples

                
    def load_classifier(self, clf_name:str, clf_type:str):
        '''
        载入分类器
        :param clf_name: 分类器名
        :param clf_type: 分类器类型

        :return clf
        '''
        models_path = os.path.join(GlobalPath.ClASS_MODEL_PATH, clf_type)
        try:
            if not os.path.exists(models_path):
                os.mkdir(models_path)
                raise FileNotFoundError(f"{models_path}模型保存路径不存在！")
            
            model_load_path = os.path.join(models_path, f"{clf_name}.pkl")
            if not os.path.exists(model_load_path):
                raise FileNotFoundError(f"指定的{clf_name}分类器模型不存在！")
            clf = load(model_load_path)
            logging.info(f"{clf_name}分类器模型载入成功")
                        
        except FileNotFoundError as err:
            logging.error("model or directory does not exist:", err)

        else:
            return clf

    
    def evaluation(self, clf_name, y_test, y_pred):
        logging.info(f"{clf_name}测试集accuracy: {round(accuracy_score(y_test, y_pred),4)}")
        logging.info(f"{clf_name}测试集precision: {round(precision_score(y_test, y_pred, pos_label=1, average='weighted'),4)}")
        logging.info(f"{clf_name}测试集F1值: {round(f1_score(y_test, y_pred, pos_label=1, average='weighted'),4)}")


