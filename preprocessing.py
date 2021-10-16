import os
from logger import Logger

import pandas as pd

LABEL_PATH = './data/labeled_data/'

lg = Logger()

# 计算特征的各种值占比
def count_share(path):
    df = pd.read_csv(path)
    attitude_triplet = []
    iters = df.iteritems()
    next(iters) # 跳过编号列

    for column, row in iters:
        attitude_triplet.append(df[column].value_counts().astype('float'))
    for triplet in attitude_triplet:
        for index, value in triplet.items():  # 计算占比
            triplet[index] = round(value/df.shape[0],3)
    
    return attitude_triplet




def clean_data():
    return    


def merge_data():
    return
