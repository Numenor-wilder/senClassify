import logging
import os 
import json
import re
import datetime

from parameter import path

import jieba
import jieba.posseg as pseg
import zhconv

RAW_TWEET=r'.\resources\raw_data'
WORD = r'.\resources\text\fenci\word_seg'

# lg = Logger()


def clean_character(sentence):
    '''
    去除无关字符
    :param sentence:
    :return:
    '''
    pattern1 = '[a-zA-Z0-9]'
    pattern2 = '\[.*?\]'
    pattern3 = re.compile(u'[^\s1234567890:：' + '\u4e00-\u9fa5]+')
    pattern4 = '[’!"#$%&\'()*+,-./:：;<=>?@[\\]^_`{|}~]+'
    line1 = re.sub(pattern1, '', sentence)  # 去除英文字母和数字
    line2 = re.sub(pattern2, '', line1)  # 去除表情
    line3 = re.sub(pattern3, '', line2)  # 去除其它字符
    line4 = re.sub(pattern4, '', line3)  # 去掉残留的冒号及其它符号
    new_sentence = ''.join(line4.split())  # 去除空白
    return new_sentence


# def is_all_chinese(strs):
#     for _char in strs:
#         if not '\u4e00' <= _char <= '\u9fa5':
#             return False
#     return True


def get_stopwords(file_path):
    stopwords = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for word in file:
            stopwords.add(word.strip('\n'))
    return stopwords


def segment(filepath, dataname, stopwords):
    in_dir = r'fenci\word_seg'
    
    seg_prefix = os.path.join(path.TEXT_PATH, in_dir)
    if not os.path.exists(seg_prefix):
        os.mkdir(seg_prefix)

    os.path.join(path.TEXT_PATH, in_dir, dataname)

    seg_list = []

    seg_path = os.path.join(seg_prefix, dataname)
    if not os.path.exists(seg_path):
        os.makedirs(seg_path)

    file = os.path.split(filepath)[-1]
    with open(filepath, 'r', encoding='utf-8') as tweets:
        logging.info(f"{file.split('.')[0]}开始分词...")
        for tw in tweets:
            js = json.loads(tw)
            if js['language'] == 'zh':
                tr_txt = zhconv.convert(js["tweet"], 'zh-tw')
                txt = clean_character(tr_txt)
                word_list = pseg.cut(txt)
                seg_list.append(word_list)

    w_filepath = os.path.join(seg_path, file.split('.')[0])
    with open(w_filepath, 'w+', encoding='utf-8') as seg_tweet:
        logging.info(f"{file.split('.')[0]}分词数据构建...")
        content = ""
        for seg in seg_list:
            for words in seg:
                if words.word not in stopwords:
                    content += words.word + ' '
            content += '\n'
        seg_tweet.write(content)
    

def pretreat(raw_data):
    jieba.set_dictionary('./resources/text/fenci/dict/traditional.txt')
    jieba.load_userdict('./resources/text/fenci/dict/user.txt')

    stopwords = get_stopwords('./resources/text/fenci/dict/stopwords.txt')

    resource_path = os.path.join(path.RAW_PATH, raw_data)
    if os.path.exists(resource_path):
        files = os.listdir(resource_path)

    jieba.enable_paddle()
    curr_time = datetime.datetime.now().strftime("%Y%m")
    data_name = raw_data + '_' + curr_time
    for filename in files:
        src_path = os.path.join(path.RAW_PATH, raw_data, filename)
        segment(src_path, data_name, stopwords)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO)
    pretreat('ww')