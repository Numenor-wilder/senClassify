import csv
import datetime
import json
import os
import pathlib

import numpy as np
import tqdm
from colorama import Fore

from common.parameters import GlobalPath


def get_max_common_range(coverage, date_range_path):
    global date_range_file
    date_range_file = date_range_path

    date_range_csv = open(date_range_path, 'r', encoding='utf-8')
    date_range = csv.reader(date_range_csv)
    next(date_range)
    data = list(date_range)
    date_min_list = np.array(data)[:, 1]
    date_max_list = np.array(data)[:, 2]

    # 找符合要求的日期区间
    date_min_list.sort()
    date_max_list.sort()
    mid = int(len(data) / 2)
    start = date_min_list[mid]
    end = date_max_list[mid]
    start_index = end_index = mid

    while start < end:
        covered = 0
        for d in data:
            # 被当前区间覆盖,即 在当前区间内有数据
            if d[1] >= start and d[2] <= end:
                covered = covered + 1
        if covered / len(data) >= coverage:
            return start, end
        # 扩张区间
        # start_date = datetime.datetime.strptime(start, "%Y-%m-%d") - datetime.timedelta(days=1)
        # end_date = datetime.datetime.strptime(end, "%Y-%m-%d") + datetime.timedelta(days=1)
        # start = start_date.strftime("%Y-%m-%d")
        # end = end_date.strftime("%Y-%m-%d")
        if start_index > 0:
            start_index = start_index - 1
            start = date_min_list[start_index]

        if end_index < len(data):
            end_index = end_index + 1
            end = date_max_list[end_index]


def cut_user_corpus(start, end, user_data_path, length=7):
    """
    按时间切分语料
    :param start: 开始日期
    :param end: 结束日期
    :param length: 切片长度(单位: 天)
    :return:
    """
    date_range_csv_file = open(date_range_file, 'r',encoding='utf-8')
    date_range_csv = csv.reader(date_range_csv_file)
    next(date_range_csv)
    date_range = list(date_range_csv)
    for d in tqdm.tqdm(date_range, desc="cutting corpus",
                    bar_format="%s{l_bar}{bar}{r_bar}%s" % (Fore.BLUE, Fore.RESET)):
        current_start = start
        current_end = end
        # 确定当前用户的查找区间
        if d[1] >= start:
            current_start = d[1]
        if d[2] <= end:
            current_end = d[2]

        # 每length天切分一次
        current_mid_date = datetime.datetime.strptime(end, "%Y-%m-%d") - datetime.timedelta(days=length)
        current_mid = current_mid_date.strftime("%Y-%m-%d")
        if current_mid < current_start:
            current_mid = current_start
        # current_output_path = os.path.join(output_cuted_path, d[0])
        # if os.path.exists(current_output_path):
        #    shutil.rmtree(current_output_path)
        # pathlib.Path(current_output_path).mkdir(parents=True, exist_ok=True)
        current_user_file = os.path.join(user_data_path, d[0] + ".json")

        # 当前要输出的文件
        file_index = 1
        # 改为 切片id/用户id.json

        dataset_name = os.path.split(user_data_path)[1]
        current_output_path = os.path.join(GlobalPath.TMP_DATA_PATH, dataset_name, str(length), str(file_index))
        pathlib.Path(current_output_path).mkdir(parents=True, exist_ok=True)
        current_output_file = os.path.join(current_output_path, d[0] + ".json")
        if os.path.exists(current_output_file) and os.path.isfile(current_output_file):
            os.remove(current_output_file)
        out_file = open(current_output_file, "w", encoding='utf-8')

        f = open(current_user_file, 'r', encoding='utf-8')
        f_lines = f.readlines()
        for line in f_lines:
            json_data = json.loads(line)
            if current_mid <= json_data['date'] <= current_end:
                # print(line)
                out_file.write(line)
            elif current_start < json_data['date'] < current_mid:
                current_end_date = datetime.datetime.strptime(current_mid, "%Y-%m-%d") - datetime.timedelta(days=1)
                current_end = current_end_date.strftime("%Y-%m-%d")
                if current_end < current_start:
                    break
                current_mid_date = datetime.datetime.strptime(current_mid, "%Y-%m-%d") - datetime.timedelta(days=length)
                current_mid = current_mid_date.strftime("%Y-%m-%d")
                if current_mid < current_start:
                    current_mid = current_start
                # 修改输出文件
                out_file.close()
                file_index = file_index + 1
                # current_output_file = os.path.join(current_output_path, str(file_index) + ".json")
                # out_file = open(current_output_file, "w", encoding='UTF8')
                # 改为 切片id/用户id.json
                current_output_path = os.path.join(GlobalPath.TMP_DATA_PATH, dataset_name, str(length), str(file_index))
                pathlib.Path(current_output_path).mkdir(parents=True, exist_ok=True)
                current_output_file = os.path.join(current_output_path, d[0] + ".json")
                out_file = open(current_output_file, "w", encoding='utf-8')
            elif json_data['date'] < current_start:
                break
    
    return os.path.join(GlobalPath.TMP_DATA_PATH, dataset_name)



# if __name__ == '__main__':
#     coverage = 0.7
#     length = 180
#     start, end = get_max_common_range(coverage)
#     start_date = datetime.datetime.strptime(start, "%Y-%m-%d")
#     end_date = datetime.datetime.strptime(end, "%Y-%m-%d")
#     print("coverage:", coverage)
#     print("start:", start, "end:", end, "diff:", end_date - start_date, "slice length:", length)
#     cut_user_corpus(start, end, length)
