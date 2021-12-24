import sys

import os 
import logging

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from joblib import dump

from fine_grained.classifier.classifier import Classifier
from common.parameters import GlobalPath


class RF_Classifier(Classifier):

    def training(self, features, labels):
        '''
        SVM分类器训练
        '''
        logging.info("随机森林分类器训练开始")
        if np.any(np.isnan(features)) or np.any(np.isnan(labels)):
            logging.warning("训练数据有空值!!!")
            return

        
        for col in labels.columns:
            rfc = RandomForestClassifier(random_state=0,max_depth=10)

            x, y = self.resample_smote(features, labels[col])

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

            rfc.fit(x_train, y_train)

            # 保存分类器
            prefix_dir = os.path.join(GlobalPath.ClASS_MODEL_PATH, 'rf')
            if not os.path.exists(prefix_dir):
                os.mkdir(prefix_dir)
            model_save_path = os.path.join(prefix_dir, f"{col}.pkl")
            dump(rfc, model_save_path)
            logging.info(f"{col}分类器模型保存成功")

            y_pred = pd.Series(rfc.predict(x_test))

            self.evaluation(col, y_test, y_pred)



    def classify(self, features, labels) ->DataFrame:
        '''
        返回值: 用户在各标签下的类别矩阵
        '''

        df_label = pd.DataFrame()    
        for label in labels:
            clf = self.load_classifier(label, 'rf')

            logging.info(f"{label}分类器标签预测")
            df_label[label] = clf.predict(features)

        return df_label
