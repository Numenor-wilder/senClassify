import os 
import json
import re
import jieba
import jieba.posseg as pseg


RAW_TWEET='./resources/raw_data/ww/'
TWEET = './resources/text/fenci/tweets'
WORD = './resources/text/fenci/word_seg'


def clean_tweet():
    path = RAW_TWEET
    txt_path = TWEET

    files = os.listdir(path)
    for file in files:
        txt = ""
        txt_list = []
        this_archive =  path + file
        with open(this_archive, 'r', encoding='utf-8') as tweets:
            for tw in tweets:
                js = json.loads(tw)
                if js['language'] == 'zh':
                    txt = clean_character(js["tweet"])
                    txt_list.append(txt)
         
        postfix = file.split('.')
        if not os.path.exists(txt_path):
            os.mkdir(txt_path)
        with open(txt_path + postfix[0] +'.txt', "w+", encoding='utf-8') as fil:
            content = '\n'.join(txt_list)
            fil.write(content)


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


def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def get_stopwords(file_path):
    stopwords = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for word in file:
            stopwords.add(word.strip('\n'))
    return stopwords


def segment(path):
    jieba.enable_paddle()
    
    in_prefix = TWEET
    out_prefix = WORD

    jieba.set_dictionary('./resources/text/fenci/dict/traditional.txt')
    jieba.load_userdict('./resources/text/fenci/dict/user.txt')

    stopwords = get_stopwords('./resources/text/fenci/dict/stopwords.txt')

    seg_list = []
    with open(in_prefix + path, 'r', encoding='utf-8') as file:
        for line in file:
            word_list = pseg.cut(line)
            seg_list.append(word_list)

    with open(out_prefix + path, 'w+', encoding='utf-8') as file:
        content = ""
        for seg in seg_list:
            for words in seg:
                if words.word not in stopwords:
                    content += words.word + ' '
        file.write(content)


def pretreat():
    # clean_tweet()
    files = os.listdir(TWEET)
    for file in files:
        segment(file)