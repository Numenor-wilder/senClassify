import re
import os
import json

RAW_TWEET='./ww/'
TWEET = './tweets_bert/'


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
                if js['language'] == 'zh' and js['retweet'] == False:
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
    pattern1 = '^[@A-Za-z0-9_\s]+'
    pattern2 = '^[^\u4e00-\u9fa5]+'
    pattern3 = re.compile(u"http\S+")
    pattern4 = re.compile(u"#\S+")
    pattern5 = '["\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"]'
    line1 = re.sub(pattern1, '', sentence)  # 去除开头@
    line2 = re.sub(pattern2, '', line1)  # 去除开头其他非汉字字符
    line3 = re.sub(pattern3, '', line2)  # 去除超链接
    line4 = re.sub(pattern4, '', line3)  # 去掉hashtag
    line5 = re.sub(pattern5, '', line4)  # 去掉表情
    new_sentence = ''.join(line5.split())  # 去除空白
    return new_sentence


if __name__ == '__main__':
    clean_tweet()