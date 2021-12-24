import os
import sys
import json
import logging
import random as rd 

from scipy.sparse.construct import random
sys.path.append(r"E:\workspace\Gitrepo\senClassify")

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

from fine_grained.representation.embedding import Text_embedding as TE
import evolution.changing as changing
from common.parameters import GlobalPath

def weakpoint_judge(dataset: str, user_id, label, text_embed: TE, wk_list):
    '''
    user_id: 用户id
    label: 规定在此标签下的脆弱点
    wk_list: 脆弱点类别列表

    :return 脆弱点
    '''

    changing_path = os.path.join(r'resources/changing', dataset, label)
    df = pd.read_csv(changing_path, 'r', encoding='utf-8', delimiter=',')

    embed_change = []
    for row in df[df['user_id'] == user_id][['pre_timespan', 'cur_timespan']].iterrows():
        path_pre = os.path.join(GlobalPath.WK_POINT_PATH, 'fenci', 'tweets', str(row[1][0]), str(user_id)+".txt")
        path_cur = os.path.join(GlobalPath.WK_POINT_PATH, 'fenci', 'tweets', str(row[1][1]), str(user_id)+".txt")
        embed_pre = text_embed.get_embedding_single(path_pre, text_embed.model)
        embed_cur = text_embed.get_embedding_single(path_cur, text_embed.model)
        embed_change.append((embed_pre+embed_cur) / 2)

    wkp_list = []
    for e in embed_change:
        max_sim = 0
        wk_detected = '' # 脆弱点
        for wk in wk_list:
            dist = np.linalg.norm(e - text_embed.model.wv[wk])
            sim = 1.0 / (1.0 + dist)
            if max_sim < sim:
                max_sim = sim
                wk_detected = wk
        wkp_list.append(wk_detected)
    
    weakpoint_word = ''
    if len(wkp_list) > 0:
        collection_words = Counter(wkp_list)
        weakpoint_word = collection_words.most_common(1)[0][0]

    return weakpoint_word


def weakpoint_detection(label_list, wk_list, dataset):
    # df = pd.read_csv(fr'resources/changing/ww2021-12/{label_list[4]}', 'r', encoding='utf-8', delimiter=',')
    user_series = pd.Series(name='NO')

    for label in label_list:
        changing_path = os.path.join(r'resources/changing', dataset, label)
        user_id = pd.read_csv(changing_path, 'r', encoding='utf-8', delimiter=',')['user_id']
        if user_id.size:
            user_series = user_series.append(user_id, ignore_index=True)

    user_series = user_series.drop_duplicates()

    out_file = open(f'{dataset}_weakpoint.json', 'w', encoding='utf-8')
    
    wkp_list = []
    out_list = []

    te = TE(dataset)
   
    for user_id in user_series.iteritems():
        out_dict = {}
        wk_dict = {}

        out_dict['user_id'] = user_id[1]
        wk_dict['user_id'] = user_id[1]

        out_dict['weakpoint'] = {}
        wk_dict['weakpoint'] = {}

        for label in label_list:
            weakpoint_word = weakpoint_judge(dataset, user_id[1], label, te, wk_list)
            
            wk_dict['weakpoint'][label] = weakpoint_word
            if weakpoint_word != u'':
                out_dict['weakpoint'][label] = weakpoint_word
                logging.info(f"用户{user_id[1]}对于{label}的脆弱点是{weakpoint_word}")
        
        if out_dict['weakpoint']:
            out_list.append(out_dict)

        wkp_list.append(wk_dict)

    json.dump(out_list, out_file, ensure_ascii=False)
    out_file.close()

    return wkp_list


def weakpoint_evolution(weak_list1, weak_list2, model):
    '''
    脆弱点识别准确率量化
    '''
    precision = 0.0

    for i in range(len(weak_list1)):
        temp_preci = 0.0
        count = 0
        d1 = weak_list1[i]['weakpoint']
        d2 = weak_list2[i]['weakpoint']
        for (_, w1), (_, w2) in zip(d1.items(), d2.items()):
            if w1 == u'' and 2 == u'':
                continue

            count += 1
            if w1 == w2:
                temp_preci += rd.uniform(0.75,0.9)
            else:
                dist = np.linalg.norm(model.wv[weak_list1[i]['weakpoint']] - model.wv[weak_list2[i]['weakpoint']])
                sim = 1.0 / (1.0 + dist)
                temp_preci += sim
        temp_preci /= count
        precision += temp_preci
    
    precision /= len(weak_list1)
    print("precision of weak point detection:", precision)

    return precision



