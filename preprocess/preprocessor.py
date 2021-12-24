import logging
import os 
import json
import re
import tqdm
from pathlib import Path

from preprocess.implicit_preproc import ImplicitExtractor as ImpExtractor
from common.parameters import GlobalPath, CustomDict
from common.utility import path_exist_operate

import jieba
import jieba.posseg as pseg
import zhconv
from colorama import Fore

class DocPreproc:

    class Segment:
        __imp = None

        def __get_imp(self):
            if not self.__imp:
                self.__imp = ImpExtractor(CustomDict.hashtag)

            return self.__imp

        def __get_stopwords(self, file_path):
            '''
            获取停用词
            '''
            stopwords = {}
            with open(file_path, 'r', encoding='utf-8') as file:
                for word in file:
                    word = word.strip()
                    if len(word):
                        stopwords[word] = word
            return stopwords

        # def __get_dictionary(self):
        #     default_dict_path = os.path.join(GlobalPath.DICT_PATH, custom_dict.default)
        #     user_dict_path = os.path.join(GlobalPath.DICT_PATH, custom_dict.user)
        #     stopwords_path = os.path.join(GlobalPath.DICT_PATH, custom_dict.stopword)

        #     jieba.set_dictionary(default_dict_path)
        #     jieba.load_userdict(user_dict_path)
        
        def __file_cut(self, file_path: Path, stopwords: dict, implicit: bool = False):
            '''
            对一个文件进行分词处理
            '''
            cuted_list = [] # 文件分词结果列表

            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text_lines = file.readlines()
            except FileNotFoundError as err:
                logging.error(err)
                return [], err
            else:
                user_id = os.path.split(file_path)[1]
                logging.info(f"{user_id.split('.')[0]}开始分词...")
                for line in text_lines:
                    tweet = json.loads(line.strip())
                    tr_txt, e = self.__json_translation(tweet, 'zh-tw')
                    if e is None:
                        if implicit:
                            imp = self.__get_imp()
                            if len(imp.features_judgement(tr_txt)) == 0:
                                continue
                        text = self.__clean_character(tr_txt)
                        word_list = pseg.cut(text)
                        word_list = [w.word for w in word_list if stopwords.get(w.word) is None]
                        cuted_list.append(word_list)
                    else:
                        continue # 跳过该行，处理下一行数据
                
                if implicit and len(cuted_list) == 0:
                    logging.info("没有提取到隐式特征...")
                
                return cuted_list, None
    

        def __file_write(self, output: str, cuted_list: list):
            seg_tweet = open(output, 'w+', encoding='utf-8')
            content = ""
            for seg_word in cuted_list:
                if len(seg_word):
                    content += ' '.join(seg_word)
                    content += '\n'
            if content:
                logging.info(f"{os.path.split(output)[-1]}分词数据构建中...")
                seg_tweet.write(content)
                seg_tweet.close()
            else:
                seg_tweet.close()
                os.remove(output)


        def __json_translation(self, json, locale):
            try:
                txt_trans = ""
                if json['language'] == 'zh':
                    txt_trans = zhconv.convert(json['tweet'], locale=locale)
            except KeyError as err:
                logging.error("数据中不含%s字段", err)
                return "", err
            else:
                return txt_trans, None


        def __clean_character(self, sentence):
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


        def segment(self, input_dataset_root: str, custom_dict: CustomDict, implicit: bool = False):
            '''
            通过文件路径指定一个数据集，推荐将文件上传到工程文件下的resources文件夹，也可以是外部绝对路径
            经过清洗和分词处理过后的结果将被输出至output_pre_path下, 与数据集文件夹同名且同结构

            参数列表
            -----
            :param input_dataset_root: 原始数据集文件夹路径
            :param output_pre_path: 输出文件夹路径
            :param custom_dict: 自定义词典类
            :param implicit: False默认, 决定是否开启隐式特征提取器

            返回值
            -----
            :return: 若返回FileNotFoundError异常，说明数据集不存在, 否则返回None
            '''
            default_dict_path = os.path.join(GlobalPath.DICT_PATH, custom_dict.default)
            user_dict_path = os.path.join(GlobalPath.DICT_PATH, custom_dict.user)
            stopwords_path = os.path.join(GlobalPath.DICT_PATH, custom_dict.stopword)

            jieba.set_dictionary(default_dict_path)
            jieba.load_userdict(user_dict_path)
            jieba.enable_paddle()
            stopwords = self.__get_stopwords(stopwords_path)


            if os.path.exists(input_dataset_root):
                for parents_dir, dirs, _ in os.walk(input_dataset_root):
                    if len(dirs):   # 没有达到最后的文件夹
                        continue
                    
                    if parents_dir == input_dataset_root:
                        output_pre_path = GlobalPath.CLEAN_DATA_NORMAL
                    else:
                        output_pre_path = GlobalPath.CLEAN_DATA_TIMESLICE


                    parents_path = Path(parents_dir)
                    file_list = os.listdir(parents_path)
                    input_pre_path, dataset_name = os.path.split(input_dataset_root)
                    if implicit:    # 区分识别隐式特征的数据集
                        dataset_name += '_implicit'
                    
                    for file in tqdm.tqdm(file_list, desc="processing dataset",
                                          bar_format="%s{l_bar}{bar}{r_bar}%s" % (Fore.BLUE, Fore.RESET)):
                        thread_path = os.path.join(parents_path, file)
                        cuted_list, e = self.__file_cut(thread_path, stopwords, implicit)

                        if e is None:
                            dataset_inner_path = parents_path.relative_to(input_pre_path)
                            
                            if implicit: 
                                inner_path_implicit = str(dataset_inner_path)
                                dir_list = inner_path_implicit.split('\\')
                                dir_list[0] = dataset_name
                                # for dir in lst:
                                dataset_inner_path = Path('\\'.join(dir_list))

                            out_path = os.path.join(output_pre_path, dataset_inner_path, file.split(".")[0])

                            path_exist_operate(out_path)    # 处理文件路径问题

                            self.__file_write(out_path, cuted_list)
                        else:
                            continue

            else:
                raise FileNotFoundError(f"{input_dataset_root}不存在")




# if __name__ == '__main__':
#     logging.basicConfig(
#         format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
#         level=logging.INFO)
