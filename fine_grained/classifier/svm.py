
import os 
import logging

from sklearn.svm import SVC
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.model_selection import train_test_split
from joblib import dump

from fine_grained.classifier.classifier import Classifier
from common.parameters import GlobalPath



class SVM_Classifier(Classifier):

    def training(self, features, labels):
        '''
        SVM分类器训练
        '''
        logging.info("SVM分类器训练开始")
        if np.any(np.isnan(features)) or np.any(np.isnan(labels)):
            logging.warning("训练数据有空值!!!")
            return

        
        for col in labels.columns:
            svm = SVC(C=1,kernel='rbf',gamma=0.1, probability=True)

            '''二分类'''
            features_bin, labels_bin = self.__sampling_to_binary(features, labels[col])

            x, y = self.resample_smote(features_bin, labels_bin)

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

            svm.fit(x_train, y_train)

            # 保存分类器
            prefix_dir = os.path.join(GlobalPath.ClASS_MODEL_PATH, 'svm')
            if not os.path.exists(prefix_dir):
                os.mkdir(prefix_dir)
            model_save_path = os.path.join(prefix_dir, f"{col}.pkl")
            dump(svm, model_save_path)

            y_pred = pd.Series(svm.predict(x_test))

            self.evaluation(col, y_test, y_pred)


    def classify(self, features, labels, probability=False) ->DataFrame:
        '''
        返回值: 用户在各标签下的类别矩阵
        '''

        df_label = pd.DataFrame()    
        for label in labels:
            clf = self.load_classifier(label, 'svm')

            if probability:
                logging.info(f"{label}分类器正样本概率预测")
                df_label[label] = clf.predict_proba(features)[:, 1]
            else:
                logging.info(f"{label}分类器标签预测")
                df_label[label] = clf.predict(features)

        return df_label


    def __sampling_to_binary(self, X: DataFrame, y: Series):
        '''
        多标签类别采样为二分类标签值
        
        :param X 样本特征值
        :param y 样本标签

        :return X_bin, y_bin 二分类数据
        '''
        index_list = []
        for index in range(y.shape[0]):
            if y[index] == 0:
                index_list.append(index)
        
        X_bin = X.drop(index_list, axis=0)
        y_bin = y.drop(index_list, axis=0)

        return X_bin, y_bin
