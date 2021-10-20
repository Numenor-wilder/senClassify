import re
import os
import json

RAW_TWEET='./resources/raw_data/ww/'
TWEET = './resources/text/tweets_bert/'


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
    
    at_pattern = "^@[A-Za-z0-9_]+ |(@[A-Za-z0-9_]+ ){2,}"
    hash_pattern = "(#[\w]+\s){2,}|#[\w\s]+$|^(#[\w]+ )|(＃[\w]+\s){2,}|＃[\w\s]+$|^(＃[\w]+ )"
    http_pattern = re.compile("http\S+")
    emoji_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)

    sentence = re.sub(http_pattern, '', sentence)  # 去除超链接
    sentence = re.sub(emoji_pattern, '', sentence)  # 去除emoji

    sentence = re.sub("\s\s(#|＃)[\w ]+((#|＃)[\w\s]+){1,}", '', sentence) # 去除#
    sentence = re.sub(hash_pattern, '', sentence) # 去除#
    sentence = re.sub(at_pattern, '', sentence) # 去除@
    
    while(re.match("^@[A-Za-z0-9_]+", sentence)):
        sentence = re.sub("^@[A-Za-z0-9_]+", r'', sentence)
    
    sentence = re.sub('[A-Za-z\s.,:!?]{5,}$|^[A-Za-z\s.,:!?]{6,}', '', sentence)

    new_sentence = ''.join(sentence.split())  # 去除首尾空白

    return new_sentence


if __name__ == '__main__':
    clean_tweet()