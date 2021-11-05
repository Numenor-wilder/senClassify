# -*- coding: utf-8 -*-
# @Author : qianj
# @Date   : 2021/11/5 15:22
# @contact: qianjinlin@mail.nwpu.edu.cn
# @Desc   : 隐式特征内容抽取器，用于细化内容筛选规则，人机共融
import logging
import re
from enum import Enum


class ImplicitExtractor(object):
    class ImplicitType(Enum):
        '''
        隐式特征枚举类型
        '''
        UNKNOWN = "unknown"
        HOT_HASHTAG = "hot_hashtag"
        NEAT_SENTENCE = "neat_sentence"

    def __init__(self):
        self.hashtags = self.__load_hashtags()  # hashtags

    def __load_hashtags(self, hashtags_file_path: str = "resources/text/fenci/dict/hot_hashtags.txt") -> list:
        '''
        载入文件，返回hashtags列表
        机器规则
            1. 这个文件根据正文hashtag统计词频构建，
            2. 有效正文多半出现频次会很多，根据此次约900人统计频次结果，抛弃低频率次数词语
                低频率次数=15，有效1831词
            3. 单个用户中出现的hashtag频次大于自身的所有hashtag50%刨除 例如筛去 例如大紀元新聞網 444 苹果日报 今日萌寵 402 等
        人工规则
            4. 剩余部分根据人工/专家知识引入，筛出无效标签，
        剩余标签：
        :param hashtags_file_path: hashtags路径默认位于./fenci/dict/hot_hashtags.txt
        :return: list of hashtags
        '''
        res = []
        try:
            file = open(hashtags_file_path, 'r', encoding='utf-8')
            for line in file.read().splitlines():
                res.append(line.split()[0])
            file.close()
            res = list(set(res))
            logging.info("hot_hashtags 载入成功")
        except IOError:
            logging.error("路径" + hashtags_file_path + "不存在 或 未指定hashtags路径，因此hashtags=[] 关于hashtags处理无效")
        return res

    def features_judgement(self, sentence: str) -> list:
        '''
        隐式特征类型判断器
        :param sentence:
        :return: list of ImplicitType
        '''
        res = []
        # 工整句式
        if len(self.neat_sentence_extractor(sentence)) != 0:
            res.append(self.ImplicitType.NEAT_SENTENCE)
        # 热点hashtags
        if len(self.hot_hashtags_extractor(sentence, self.hashtags)) != 0:
            res.append(self.ImplicitType.HOT_HASHTAG)
        return res

    def hot_hashtags_extractor(self, content: str, hot_hashtags: list) -> list:
        '''
        传入一句话，能够识别出该句话中是否包含重点hashtags，其中hot hashtags应该由一组list传入
        :param content: 传入的句子
        :param hot_hashtags: hot_hashtags列表
        :return: 包含的整句话中的 list of hashtags
        '''
        res = []
        for hashtag in hot_hashtags:
            if hashtag in content:
                res.append(hashtag)
        return res

    def neat_sentence_extractor(self, content: str) -> str:
        '''
        传入一句话，能够识别出工整句式的部分，如果存在工整句式，那么返回工整句式。
        例如：
        输入：我想在这儿说点话：一栋大楼，别看它貌似坚固，若是基梁烂透，则说倒就倒！一家公司，别看它家大业大，若有一行不慎，则说垮就垮！一个政权，别看它兵痞警黑，若已民怨沸腾，则说亡就亡！
        输出：一栋大楼，别看它貌似坚固，若是基梁烂透，则说倒就倒！一家公司，别看它家大业大，若有一行不慎，则说垮就垮！一个政权，别看它兵痞警黑，若已民怨沸腾，则说亡就亡！
        输入：今天天气真好
        输出：‘’
        输入：一朝风云散尽时，昨日红宴今日哀！
        输出：一朝风云散尽时，昨日红宴今日哀！
        :param content:输入一段话
        :return:输出工整句式，不存在则返回‘’
        '''
        # 先处理冒号 ‘：|:’ 前的部分
        if content.count('：') + content.count(':') > 0:
            rs = re.split('[：:]', content)
            for r in rs:
                result = self.neat_sentence_extractor(r)
                if result:
                    return result
            return ""
        else:
            raw_raw_array = re.split('[,，!！。.?？]', content)  # 分割语句
            raw_array = []
            for s in raw_raw_array:
                if s and not s.isspace():  # 去除空白
                    raw_array.append(s)
            if len(raw_array) == 0:
                return ""
            start_index = 0  # 工整句开始位置
            last_start_index = 0  # 最后一次比较的s1位置
            end_index = 0  # 工整句结束位置
            while start_index + 1 < len(raw_array):
                end_index = start_index
                # 步长
                step_len = 1
                while step_len <= len(raw_array) / 2:  # 增加步长
                    i = 0
                    while i + step_len < len(raw_array):
                        s1 = re.sub('[^\u4e00-\u9fa5]+', '', raw_array[i])  # 要比较的句子1，只保留中文
                        s2 = re.sub('[^\u4e00-\u9fa5]+', '', raw_array[i + step_len])  # 要比较的句子2
                        if abs(len(s1) - len(s2)) > 0:  # 允许句子之间相差0个字也为工整句式
                            break
                        last_start_index = i
                        end_index = i + step_len
                        i += 1
                    step_len += 1
                if start_index != end_index:
                    if step_len != 2 and start_index == last_start_index:
                        break
                    # 找到排比子句
                    if (end_index - start_index + 1) % 2 != 0:
                        end_index -= 1
                    cut_start = content.find(raw_array[start_index])
                    cut_end = content.find(raw_array[end_index]) + len(raw_array[end_index])

                    while cut_end < len(content):
                        if '\u4e00' <= content[cut_end] <= '\u9fa5':
                            cut_end -= 1
                            break
                        cut_end += 1
                    return content[cut_start:cut_end + 1]
                start_index += 1
            return ""


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO)
    imp = ImplicitExtractor()