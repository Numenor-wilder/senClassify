# -*- coding: utf-8 -*-
# @Author : qianj
# @Date   : 2021/12/17 0:28
# @contact: qianjinlin@mail.nwpu.edu.cn
# @Desc   : TODO
import csv
import json
import os


class Evolution_Pattern:
    def __init__(self):
        pass

    def __get_timespan_cnt(self, probability_dir: str) -> int:
        '''
        得到指定目录下时间片的个数
        :param probability_dir: 概率文件夹路径，例如./resources/text/weakpoint/probability
        :return: int数字，timespan最大值
        '''
        file_list = os.listdir(probability_dir)
        for file_name in file_list:
            file_path = os.path.join(probability_dir, file_name)
            input_file = open(file_path, 'r', encoding='utf-8')
            reader = csv.reader(input_file)
            for line in reader:
                return len(line) - 1
        return 0


    def __get_time_line_labels_from_dir(self, probability_dir: str, user_id: str) -> dict:
        '''
        得到用户在指定各个标签结果
        例如:
        {'id': '1000423836775428096', 'probability_timespan': {'1': {'一中各表': 0.8996983182550596, '一国两制': 0.11045881251316371, '中共': 0.02775275900181325, '中华民国': 0.7588496334473097, '中国': 0.06758689964416095, '中国人身份': 0.061488729770036755, '华人身份': 0.9877697924927213, '华语': 0.9999985801829021, '国民党': 0.013505741437330757, '对大陆宣称': 0.07031325953218107, '民主政府': 0.852846995398387, '民进党': 0.12637751537611963, '节日': 0.9999999993359133, '饮食习惯': 0.9752606238981733}, '2': {'一中各表': 0.8980382781809199, '一国两制': 0.11166217638229771, '中共': 0.02784834648269997, '中华民国': 0.7581966053523818, '中国': 0.06998802302786641, '中国人身份': 0.06242735632134098, '华人身份': 0.987668839720065, '华语': 0.99999887534385, '国民党': 0.013643316496651969, '对大陆宣称': 0.06410399268334165, '民主政府': 0.8530662752308299, '民进党': 0.12892033495955596, '节日': 0.9999999993604073, '饮食习惯': 0.9771680813001127}, '3': {'一中各表': 0.0, '一国两制': 0.0, '中共': 0.0, '中华民国': 0.0, '中国': 0.0, '中国人身份': 0.0, '华人身份': 0.0, '华语': 0.0, '国民党': 0.0, '对大陆宣称': 0.0, '民主政府': 0.0, '民进党': 0.0, '节日': 0.0, '饮食习惯': 0.0}, '4': {'一中各表': 0.0, '一国两制': 0.0, '中共': 0.0, '中华民国': 0.0, '中国': 0.0, '中国人身份': 0.0, '华人身份': 0.0, '华语': 0.0, '国民党': 0.0, '对大陆宣称': 0.0, '民主政府': 0.0, '民进党': 0.0, '节日': 0.0, '饮食习惯': 0.0}, '5': {'一中各表': 0.0, '一国两制': 0.0, '中共': 0.0, '中华民国': 0.0, '中国': 0.0, '中国人身份': 0.0, '华人身份': 0.0, '华语': 0.0, '国民党': 0.0, '对大陆宣称': 0.0, '民主政府': 0.0, '民进党': 0.0, '节日': 0.0, '饮食习惯': 0.0}, '6': {'一中各表': 0.0, '一国两制': 0.0, '中共': 0.0, '中华民国': 0.0, '中国': 0.0, '中国人身份': 0.0, '华人身份': 0.0, '华语': 0.0, '国民党': 0.0, '对大陆宣称': 0.0, '民主政府': 0.0, '民进党': 0.0, '节日': 0.0, '饮食习惯': 0.0}, '7': {'一中各表': 0.8952388566502882, '一国两制': 0.11317252206258505, '中共': 0.028411498348225497, '中华民国': 0.7569475291874362, '中国': 0.07200645527408457, '中国人身份': 0.06200077050616517, '华人身份': 0.9873164829985649, '华语': 0.9999992257492375, '国民党': 0.013794888110083904, '对大陆宣称': 0.05938269783728361, '民主政府': 0.8538975058077083, '民进党': 0.13224086271788843, '节日': 0.9999999994064808, '饮食习惯': 0.9781505192701879}, '8': {'一中各表': 0.8944480942127607, '一国两制': 0.1133861342311568, '中共': 0.028570474411890767, '中华民国': 0.7566960507948413, '中国': 0.07256612175965144, '中国人身份': 0.062402422918500004, '华人身份': 0.9872572425379015, '华语': 0.99999925053883, '国民党': 0.013722144935758343, '对大陆宣称': 0.05947463830773528, '民主政府': 0.8539347315102988, '民进党': 0.13174134653147457, '节日': 0.9999999994234094, '饮食习惯': 0.9785993763416662}, '9': {'一中各表': 0.893924315653674, '一国两制': 0.1135936327172228, '中共': 0.02862335838399997, '中华民国': 0.7567025597797246, '中国': 0.07328539537314804, '中国人身份': 0.06254983774471107, '华人身份': 0.9871052484522493, '华语': 0.9999992938338059, '国民党': 0.013777756031483249, '对大陆宣称': 0.06097828866152901, '民主政府': 0.8541762807280466, '民进党': 0.13209816119788367, '节日': 0.9999999994180536, '饮食习惯': 0.9789045606928181}, '10': {'一中各表': 0.8936270370518983, '一国两制': 0.11360926622150713, '中共': 0.028577300019031217, '中华民国': 0.7568236731038466, '中国': 0.07370571095016422, '中国人身份': 0.06240569612403337, '华人身份': 0.9870345745417479, '华语': 0.9999992929321725, '国民党': 0.013977814283348901, '对大陆宣称': 0.05977990706694258, '民主政府': 0.8542739129169629, '民进党': 0.13264898153287139, '节日': 0.999999999414207, '饮食习惯': 0.9790777001999487}, '11': {'一中各表': 0.0, '一国两制': 0.0, '中共': 0.0, '中华民国': 0.0, '中国': 0.0, '中国人身份': 0.0, '华人身份': 0.0, '华语': 0.0, '国民党': 0.0, '对大陆宣称': 0.0, '民主政府': 0.0, '民进党': 0.0, '节日': 0.0, '饮食习惯': 0.0}, '12': {'一中各表': 0.0, '一国两制': 0.0, '中共': 0.0, '中华民国': 0.0, '中国': 0.0, '中国人身份': 0.0, '华人身份': 0.0, '华语': 0.0, '国民党': 0.0, '对大陆宣称': 0.0, '民主政府': 0.0, '民进党': 0.0, '节日': 0.0, '饮食习惯': 0.0}, '13': {'一中各表': 0.0, '一国两制': 0.0, '中共': 0.0, '中华民国': 0.0, '中国': 0.0, '中国人身份': 0.0, '华人身份': 0.0, '华语': 0.0, '国民党': 0.0, '对大陆宣称': 0.0, '民主政府': 0.0, '民进党': 0.0, '节日': 0.0, '饮食习惯': 0.0}, '14': {'一中各表': 0.0, '一国两制': 0.0, '中共': 0.0, '中华民国': 0.0, '中国': 0.0, '中国人身份': 0.0, '华人身份': 0.0, '华语': 0.0, '国民党': 0.0, '对大陆宣称': 0.0, '民主政府': 0.0, '民进党': 0.0, '节日': 0.0, '饮食习惯': 0.0}, '15': {'一中各表': 0.0, '一国两制': 0.0, '中共': 0.0, '中华民国': 0.0, '中国': 0.0, '中国人身份': 0.0, '华人身份': 0.0, '华语': 0.0, '国民党': 0.0, '对大陆宣称': 0.0, '民主政府': 0.0, '民进党': 0.0, '节日': 0.0, '饮食习惯': 0.0}, '16': {'一中各表': 0.0, '一国两制': 0.0, '中共': 0.0, '中华民国': 0.0, '中国': 0.0, '中国人身份': 0.0, '华人身份': 0.0, '华语': 0.0, '国民党': 0.0, '对大陆宣称': 0.0, '民主政府': 0.0, '民进党': 0.0, '节日': 0.0, '饮食习惯': 0.0}, '17': {'一中各表': 0.0, '一国两制': 0.0, '中共': 0.0, '中华民国': 0.0, '中国': 0.0, '中国人身份': 0.0, '华人身份': 0.0, '华语': 0.0, '国民党': 0.0, '对大陆宣称': 0.0, '民主政府': 0.0, '民进党': 0.0, '节日': 0.0, '饮食习惯': 0.0}, '18': {'一中各表': 0.0, '一国两制': 0.0, '中共': 0.0, '中华民国': 0.0, '中国': 0.0, '中国人身份': 0.0, '华人身份': 0.0, '华语': 0.0, '国民党': 0.0, '对大陆宣称': 0.0, '民主政府': 0.0, '民进党': 0.0, '节日': 0.0, '饮食习惯': 0.0}, '19': {'一中各表': 0.0, '一国两制': 0.0, '中共': 0.0, '中华民国': 0.0, '中国': 0.0, '中国人身份': 0.0, '华人身份': 0.0, '华语': 0.0, '国民党': 0.0, '对大陆宣称': 0.0, '民主政府': 0.0, '民进党': 0.0, '节日': 0.0, '饮食习惯': 0.0}, '20': {'一中各表': 0.0, '一国两制': 0.0, '中共': 0.0, '中华民国': 0.0, '中国': 0.0, '中国人身份': 0.0, '华人身份': 0.0, '华语': 0.0, '国民党': 0.0, '对大陆宣称': 0.0, '民主政府': 0.0, '民进党': 0.0, '节日': 0.0, '饮食习惯': 0.0}, '21': {'一中各表': 0.0, '一国两制': 0.0, '中共': 0.0, '中华民国': 0.0, '中国': 0.0, '中国人身份': 0.0, '华人身份': 0.0, '华语': 0.0, '国民党': 0.0, '对大陆宣称': 0.0, '民主政府': 0.0, '民进党': 0.0, '节日': 0.0, '饮食习惯': 0.0}, '22': {'一中各表': 0.0, '一国两制': 0.0, '中共': 0.0, '中华民国': 0.0, '中国': 0.0, '中国人身份': 0.0, '华人身份': 0.0, '华语': 0.0, '国民党': 0.0, '对大陆宣称': 0.0, '民主政府': 0.0, '民进党': 0.0, '节日': 0.0, '饮食习惯': 0.0}, '23': {'一中各表': 0.0, '一国两制': 0.0, '中共': 0.0, '中华民国': 0.0, '中国': 0.0, '中国人身份': 0.0, '华人身份': 0.0, '华语': 0.0, '国民党': 0.0, '对大陆宣称': 0.0, '民主政府': 0.0, '民进党': 0.0, '节日': 0.0, '饮食习惯': 0.0}, '24': {'一中各表': 0.0, '一国两制': 0.0, '中共': 0.0, '中华民国': 0.0, '中国': 0.0, '中国人身份': 0.0, '华人身份': 0.0, '华语': 0.0, '国民党': 0.0, '对大陆宣称': 0.0, '民主政府': 0.0, '民进党': 0.0, '节日': 0.0, '饮食习惯': 0.0}, '25': {'一中各表': 0.0, '一国两制': 0.0, '中共': 0.0, '中华民国': 0.0, '中国': 0.0, '中国人身份': 0.0, '华人身份': 0.0, '华语': 0.0, '国民党': 0.0, '对大陆宣称': 0.0, '民主政府': 0.0, '民进党': 0.0, '节日': 0.0, '饮食习惯': 0.0}}}
        :param probability_dir: 概率文件夹路径，例如./resources/text/weakpoint/probability
        :param user_id: 需要得到的用户的id
        :return: dict形式的结果
        '''
        timespan_cnt = self.__get_timespan_cnt(probability_dir)

        result_dict = {'id': user_id}
        file_list = os.listdir(probability_dir)
        for file_name in file_list:
            file_path = os.path.join(probability_dir, file_name)
            input_file = open(file_path, 'r', encoding='utf-8')
            reader = csv.DictReader(input_file)
            for line in reader:
                if line['NO'] == user_id:
                    if not result_dict.get('probability_timespan'):
                        result_dict['probability_timespan'] = {}
                    for timespan in range(1, timespan_cnt + 1):
                        timespan = str(timespan)
                        if not result_dict['probability_timespan'].get(timespan):
                            result_dict['probability_timespan'][timespan] = {}
                        result_dict['probability_timespan'][timespan][file_name] = float(line[timespan])
            input_file.close()
        return result_dict


    def __get_new_simliar_probility_dict(self, vector: list) -> dict:
        """
        传入一个向量，得到类似下方结果
        {'camp': '泛绿', 'prob': 0.5345224838248487, 'prob_list': {'铁红': -0.5050762722761054, '泛红': -0.47809144373375745, '铁蓝': 0.2182178902359924, '泛蓝': 0.2519763153394848, '泛绿': 0.5345224838248487, '铁绿': 0.4364357804719848}}
        :param vector: 传入的向量值，其中vector为一个列表例如 [0.9999985801829021,0.9752606238981733,0.9999999993359133,0.9877697924927213,0.061488729770036755,0.06758689964416095,0.02775275900181325,]
        :return: 结果值dict
        """
        # | 华语 | 饮食习惯 | 节日 | 华人身份 | 中国人身份 | 中国 | 中共 | 一国两制 | 民主政府 | 中华民国 | 国民党 | 民进党 | 一中各表 | 对大陆宣称 |
        NEW_STANDARD_CAMP = [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # 铁红
                            [1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0, 0],  # 泛红
                            [1, 1, 1, 1, 1, 1, 0, 0, 0.5, 1, 1, 0.5, 1, 1],  # 铁蓝
                            [1, 1, 1, 1, 1, 0.5, 0.5, 0, 1, 1, 1, 0.5, 0.5, 0.5],  # 泛蓝
                            [1, 1, 1, 1, 0, 0.5, 0.5, 0, 1, 0.5, 0.5, 1, 0.5, 0.5],  # 泛绿
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0.5, 0.5]]  # 铁绿

        CAMP_RESULT = ['铁红', '泛红', '铁蓝', '泛蓝', '泛绿', '铁绿']

        result = {'camp': '未知', 'prob': -1, 'prob_list': {}}

        max_prob = -1
        for index in range(len(NEW_STANDARD_CAMP)):
            cur_camp_name = CAMP_RESULT[index]
            cur_stamdard_camp = NEW_STANDARD_CAMP[index]
            # 相似度计算Cosine Similarity
            import numpy as np
            np.seterr(invalid='ignore')
            a = np.array(vector)
            b = np.array(cur_stamdard_camp)
            ma = np.linalg.norm(a)
            mb = np.linalg.norm(b)
            sim = (np.matmul(a, b)) / (ma * mb)
            # 结果记录
            result['prob_list'][cur_camp_name] = sim
            if sim > max_prob:
                max_prob = sim
                result['camp'] = CAMP_RESULT[index]
                result['prob'] = sim
        return result


    def __generate_vector_timespan(self, user_dict: dict) -> dict:
        '''
            得到用户在指定各个标签结果
            例如:
            :param probability_dir: 概率文件夹路径，例如./resources/text/weakpoint/probability
            :param user_id: 需要得到的用户的id
            :return: dict形式的结果
        '''
        result_dict = {'id': user_dict['id'], 'camp_timespan': {},
                    'probability_timespan': user_dict['probability_timespan']}
        # 生成标准向量
        LABEL_NAME = ['华语', '饮食习惯', '节日', '华人身份', '中国人身份', '中国', '中共', '一国两制', '民主政府', '中华民国', '国民党', '民进党', '一中各表',
                    '对大陆宣称']
        timespan_cnt = len(user_dict['probability_timespan'])
        for timespan in range(1, timespan_cnt + 1):
            timespan = str(timespan)
            cur_timespan_dict = user_dict['probability_timespan'][timespan]
            vector = []
            # 生成存储在了vector中，供后续调用
            for label in LABEL_NAME:
                vector.append(cur_timespan_dict[label])
            # 得到每个时间片camp结果
            result = self.__get_new_simliar_probility_dict(vector)
            # 结果存储
            result_dict['camp_timespan'][timespan] = result
        return result_dict


    def __get_user_list(self, probability_dir: str) -> list:
        '''
        得到用户名单
        :param probability_dir: 概率文件夹路径，例如./resources/text/weakpoint/probability
        :return: user_list
        '''
        user_list = []
        file_list = os.listdir(probability_dir)
        for file_name in file_list:
            file_path = os.path.join(probability_dir, file_name)
            input_file = open(file_path, 'r', encoding='utf-8')
            reader = csv.DictReader(input_file)
            for line in reader:
                user_list.append(line['NO'])
            input_file.close()
            return user_list
        return user_list



    def __store_anly_json(self, json_dict):
        user_id = json_dict['id']
        camp_timespan = json_dict['camp_timespan']
        probability_timespan = json_dict['probability_timespan']
        length = len(camp_timespan)
        # 存储数据
        user_store = []
        for timespan in range(1, length + 1):
            timespan = str(timespan)
            camp = camp_timespan[timespan]['camp']
            prob = probability_timespan[timespan]
            if camp != '未知':
                user_store.append([user_id, timespan, camp, prob])
        return user_store


    def __anly_json(self, store_list, output_file_path):
        # store_list: [user_id, timespan, camp, prob]
        last_list = store_list[0]  # 上个camp等信息
        file = open(output_file_path, 'a', encoding='utf-8')
        for i in range(1, len(store_list)):
            lastcamp = last_list[2]
            cur_list = store_list[i]
            curcamp = cur_list[2]

            if lastcamp != curcamp:
                lastprob = last_list[3]
                curprob = cur_list[3]
                # 情感变化
                change_reason = [store_list[0][0]]
                for key in lastprob.keys():
                    if lastprob[key] - curprob[key] > 0.5:
                        change_reason.append(key + "-")
                    elif curprob[key] - lastprob[key] > 0.5:
                        change_reason.append(key + "+")
                temp_dict = {}
                temp_dict['pre_camp'] = str(lastcamp)
                temp_dict['cur_camp'] = str(curcamp)
                temp_dict['user_id'] = change_reason[0]
                temp_dict['reason'] = change_reason[1:]
                temp_dict['pre_timespan'] = i
                temp_dict['cur_timespan'] = i + 1

                if len(change_reason) > 1:
                    json.dump(temp_dict, file, ensure_ascii=False)
                    file.write("\n")
                    print(temp_dict)
                last_list = cur_list
        file.close()


    def getjsonlist(self, probability_dir, output_file_path):
        '''
        主调，将指定路径文件夹下文件分析生成阵营文件
        :param probability_dir:
        :param output_file_path:
        :return:
        '''
        json_list = []
        user_list = self.__get_user_list(probability_dir)
        for user_id in user_list:
            result_dict = self.__get_time_line_labels_from_dir(probability_dir, user_id)
            camp_dict = self.__generate_vector_timespan(result_dict)
            store = self.__store_anly_json(camp_dict)
            self.__anly_json(store, output_file_path)
            json_list.append(camp_dict)
            # input()
        return json_list


# if __name__ == '__main__':
#     json_list = getjsonlist('./probability', 'change.json')
