import csv
import json


class Disting_Camp:
    '''
    细粒度划分政治阵营
    '''
    def get_simliar_probility_dict(self, vector: list) -> dict:
        """
        传入一个向量，得到类似下方结果
        {'camp': '泛绿', 'prob': 0.5345224838248487, 'prob_list': {'铁红': -0.5050762722761054, '泛红': -0.47809144373375745, '铁蓝': 0.2182178902359924, '泛蓝': 0.2519763153394848, '泛绿': 0.5345224838248487, '铁绿': 0.4364357804719848}}
        :param vector: 传入的向量值，其中vector为一个列表例如 [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
        :return: 结果值dict
        """
        # | 华语 | 饮食习惯 | 节日 | 华人身份 | 中国人身份 | 中国 | 中共 | 一国两制 | 民主政府 | 中华民国 | 国民党 | 民进党 | 一中各表 | 对大陆宣称 |
        STANDARD_CAMP = [[1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1],  # 铁红
                        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, -1, -1],  # 泛红
                        [1, 1, 1, 1, 1, 1, -1, -1, 0, 1, 1, 0, 1, 1],  # 铁蓝
                        [1, 1, 1, 1, 1, 0, 0, -1, 1, 1, 1, 0, 0, 0],  # 泛蓝
                        [1, 1, 1, 1, -1, 0, 0, -1, 1, 0, 0, 1, 0, 0],  # 泛绿
                        [-1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 0, 0]]  # 铁绿
        CAMP_RESULT = ['铁红', '泛红', '铁蓝', '泛蓝', '泛绿', '铁绿']

        result = {'camp': '未知', 'prob': -1, 'prob_list': {}}
        max_prob = -1
        for index in range(len(STANDARD_CAMP)):
            cur_camp_name = CAMP_RESULT[index]
            cur_stamdard_camp = STANDARD_CAMP[index]
            # 相似度计算Cosine Similarity
            import numpy as np
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


    def generate_camp_from_label(self, input_file_path, output_file_path):
        '''
        根据label生成阵营
        :param input_file_path: 细粒度标签文件
        :param output_file_path: 输出的阵营信息json保存
        :return:
        '''

        input_file = open(input_file_path, 'r', encoding='utf-8')
        reader = csv.reader(input_file)
        next(reader)

        output_file = open(output_file_path, 'w', encoding='utf-8')

        for line in reader:
            numbers = line[1:]
            numbers = [int(x) for x in numbers]
            dict_res = self.get_simliar_probility_dict(numbers)
            dict_res['id'] = line[0]
            print(dict_res)
            json.dump(dict_res, output_file, ensure_ascii=False)
            output_file.write("\n")
        output_file.close()


if __name__ == '__main__':
    pass
    # generate_camp_from_label_file("data/old_all_features.csv", 'output/camp_label.json')
