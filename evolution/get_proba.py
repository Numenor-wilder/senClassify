import sys
sys.path.append(r"E:\workspace\Gitrepo\senClassify")

import os
import logging

import pandas as pd
import numpy as np

from fine_grained.representation.embedding import Text_embedding
from fine_grained.classifier.svm import SVM_Classifier
from common.parameters import GlobalPath 
from common.utility import check_multi_dir, create_multi_dir


# user_date_range_file = os.path.join(GlobalPath.WK_POINT_PATH, "user_date_range.csv")
# user_twoClasses_label_data_file = os.path.join(GlobalPath.WK_POINT_PATH, "user_twoClasses_label_data.csv")
# user_twoClasses_label_probability_file = os.path.join(GlobalPath.WK_POINT_PATH, "user_twoClasses_label_probability.csv")


class Proba():
    def __init__(self, doc_path, label_path):

        self.doc_path = doc_path
        self.label_path = label_path


    def get_time_slices(self, time_slices_path):
        '''
        获取时间片文件路径列表
        '''
        
        dirs = []
        time_slice_sn = os.listdir(time_slices_path)
        time_slice_sn.sort(key=int)
        for sn in time_slice_sn:
            path = os.path.join(time_slices_path, sn)
            dirs.append(path)
        return dirs


    def proba_predict(self, doc_path, label_path) -> dict:
        '''
        返回用户在各个标签下为正样本的概率矩阵字典，key为时间片编号
        '''
        # label_path = r'.\data\labeled_data\old_all_features.csv'
        labels_vector = pd.read_csv(label_path, dtype={'NO': 'str'}).sort_values('NO')
        labels = labels_vector.iloc[:,1:].columns

        time_slice_matrix = {}
        time_slices_path = os.path.join(GlobalPath.WK_POINT_PATH, 'fenci', 'tweets')
        slices = self.get_time_slices(time_slices_path)

        te = Text_embedding(doc_path)
        for slice_pre in slices:
            user_list = []
            feature_list = []
            slice_files = os.listdir(slice_pre)
            logging.info(f"分析第{os.path.split(slice_pre)[-1]}号时间片...")
            for userid_file in slice_files:
                time_slice_path = os.path.join(slice_pre, userid_file)
                logging.info(f"计算{userid_file}文件文本表示...")
                feature_list.append(te.get_embedding_single(time_slice_path, te.model)) 
                user_list.append(userid_file.split('.')[0])

            user_id = pd.Series(user_list, name='NO')
            features = pd.DataFrame(feature_list)
            features.fillna(0, inplace=True)
            
            clf = SVM_Classifier()

            proba_matrix = clf.classify(features, labels, probability=True)
            
            # 时间片内，用户在各个标签下为正样本的概率矩阵
            time_slice_matrix[os.path.split(slice_pre)[-1]] = pd.concat([user_id, proba_matrix],axis=1)

        return time_slice_matrix


    def generate_label_matrix(self, doc_path, time_slice_matrix):
        matrixs = list(time_slice_matrix.values())
        labels = matrixs[0].columns[1:]
        cols = list(time_slice_matrix.keys())
        
        label_proba_dict = {}

        for label in labels:
            label_df = matrixs[0][['NO', label]].set_index('NO') # 以首个matrix构造初始datafram
            proba_matrixs = [proba_matrix for _, proba_matrix in time_slice_matrix.items()]
            
            for matrix in proba_matrixs[1:]: # 循环跳过第一项
                label_df = pd.concat([label_df, matrix[['NO', label]].set_index('NO')], axis=1)

            label_df.fillna(0, inplace=True)
            label_df.columns = cols
            label_df.insert(0, 'NO', label_df.index)
            
            dataset_name = os.path.split(doc_path)[-1]
            csv_path = os.path.join(GlobalPath.WK_POINT_PATH, 'probability', 'targetTwDoc-1', label)

            multi_dir, dirs = check_multi_dir(csv_path)
            if dirs:
                create_multi_dir(multi_dir[0], dirs[0])
                
            label_df.to_csv(csv_path, mode='w', encoding='utf-8', index=False)

            label_proba_dict[label] = label_df

        return label_proba_dict



# if __name__ == '__main__':
#     logging.basicConfig(
#         format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
#         level=logging.INFO)

#     doc_path = r'fenci\word_seg\ww2021-10'
#     label_path = r'data\labeled_data\old_all_features.csv'
#     df = proba_predict(doc_path, label_path)
#     # print(df)
#     label_matrix = generate_label_matrix(df)
#     # print(label_matrix)