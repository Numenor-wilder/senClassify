import sys
sys.path.append("/home/work/code")

import os
import shutil

import pandas as pd
import numpy as np

import logging

from common.parameters import GlobalPath, CustomDict
from common.parameters import EmbeddingHyperParma as EmbHparam

from preprocess.preprocessor import DocPreproc as DP
from timeslic import time_slicing

from fine_grained.representation.tfidf import Tfidf_Embedding as TE
from fine_grained.classifier.rf import RF_Classifier as RFC
from fine_grained.classifier.svm import SVM_Classifier as SVC
from fine_grained.distingCamp import Disting_Camp

from evolution.get_proba import Proba
from evolution.changing import export_result_to_csv_timespan_and_attitude_from_csvfile
from evolution.extract_patter import Evolution_Pattern as EP
from evolution.weakpoint import weakpoint_detection, weakpoint_evolution

dataset = 'ww'

def data_preproc(dataset_path: str, implicit: bool = False):
    '''
    常规的数据预处理，包括筛选，清洗，分词等操作
    :param dataset_path: 需要被处理的数据集相对路径
    :param implicit: 隐式特征识别器开启标识符
    '''
    seg = DP.Segment()
    seg.segment(dataset_path, CustomDict, implicit)


def time_sliced_preproc(dataset_path: str, coverage: float, length: int):
    '''
    时间分片的数据预处理，在按照参数要求分割时间片的基础上进行数据清洗和分词等操作
    :param dataset_path: 需要被处理的数据集相对路径
    :param coverage: 数据的覆盖范围, 取值为0~1之间
    :param length: 被分片的时间片长度, 单位为天
    '''
    sliced_dataset = time_slicing(dataset_path, coverage, length)
    seg = DP.Segment()
    seg.segment(sliced_dataset, GlobalPath.CLEAN_DATA_TIMESLICE, CustomDict, False)
    shutil.rmtree(sliced_dataset)


def embedding(dataset):
    te = TE(dataset)

    return te.get_tfidf_embedding(dataset)


def classify(dataset, input_embed, input_label):
    rfc = RFC()
    Xs, Ys, user_id = rfc.load_training_data(input_embed, input_label)
    multi_label = rfc.classify(Xs, Ys.columns)
    fine_label = pd.concat([user_id, multi_label],axis=1)
    fine_label.to_csv(f'{dataset}_fine_label.csv', mode='w', encoding='utf-8', index=False)

    return fine_label


def bin_classify(dataset, input_embed, input_label):
    svc = SVC()
    Xs, Ys, user_id = svc.load_training_data(input_embed, input_label)
    multi_label = svc.classify(Xs, Ys.columns, probability=True)
    fine_label = pd.concat([user_id, multi_label],axis=1)
    fine_label.to_csv(f'{dataset}_fine_proba.csv', mode='w', encoding='utf-8', index=False)

    return fine_label


def disCamp(input_path, out_path):
    dc = Disting_Camp()
    dc.generate_camp_from_label(input_path, out_path)



def get_slice_proba(doc_path, label_path):
    proba = Proba(doc_path, label_path)

    df = proba.proba_predict(proba.doc_path, proba.label_path)
    proba.generate_label_matrix(proba.doc_path, df)



def changing(proba_dir, change_dir):
    # proba_dir = r'resources\text\weakpoint\probability'
    file_list = os.listdir(proba_dir)

    for file in file_list:
        input_file = os.path.join(proba_dir, file)
        output_file = os.path.join(change_dir, file)
        # output_file = fr'resources/changing/{file}'
       
        # multi_dir, dirs = check_multi_dir(input_file, output_file)
        # if dirs:
        #     create_multi_dir(multi_dir, dirs)

        export_result_to_csv_timespan_and_attitude_from_csvfile(input_file, output_file)


def evolution(doc_path, out_put):
    ep = EP()
    ep.getjsonlist(fr'resources\text\weakpoint\probability\{doc_path}', out_put)


def weakpoint(dataset):
    te = TE(dataset)

    label_list = ['华语','饮食习惯','节日','华人身份','中国人身份','中国','中共','一国两制','民主政府','中华民国','国民党','民进党','一中各表','对大陆宣称']
    wk_list = ['亲情','声誉','金钱','权力','爱情','事业', '自由', '舆论', '人格尊严', '人身安全']

    weak_list1 = weakpoint_detection(label_list, wk_list, dataset)
    weak_list2 = weakpoint_detection(label_list, wk_list, dataset)

    weakpoint_evolution(weak_list1, weak_list2, te.model)



if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO)
    
    raw_data_path = os.path.join(GlobalPath.RAW_DATA_PATH, dataset)
    data_preproc(raw_data_path)
    # time_sliced_preproc(os.path.join(GlobalPath.RAW_DATA_PATH, dataset), 0.7, 90)
  
    # df = embedding(dataset)

    # TE(dataset)

    # classify('targetTwDoc-1_d512_e5_w10', 'old_all_features.csv')
    # bin_classify('ww2021-12_d512_e5_w10', 'old_all_features.csv')

    # disCamp('targetTwDoc-1_fine_label.csv', 'targetTwDoc-1_d512_e5_w10_political_camp.json')

    #-------------------------------------分界线----------------------------------

    # date_common_range()

    # segment()

    # cut_preproc()

    # get_slice_proba(r'fenci\word_seg\targetTwDoc2021-12', r'data\labeled_data\old_all_features.csv')

    # changing(r'resources\text\weakpoint\probability\targetTwDoc-1', r'resources\changing')

    # evolution('targetTwDoc-1','targetTwDoc-1_evolution_pattern.json')

    # weakpoint('ww2021-12')
    

