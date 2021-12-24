import csv
import datetime
import json
import os

import tqdm
from colorama import Fore

from common.parameters import GlobalPath

def get_date_range(raw_data_path: str):
    '''
    raw_data_path: 输入路径

    :return date_range_file: 返回临时时间范围文件路径
    '''

    data_files = []
    for f in os.listdir(raw_data_path):
        file_path = os.path.join(raw_data_path, f)
        if os.path.isfile(file_path) and os.path.splitext(f)[1] == '.json':
            data_files.append(file_path)

    dataset = os.path.split(raw_data_path)[1]

    date_range_file = os.path.join(GlobalPath.TMP_DATA_PATH, f'{dataset}_date_range.csv')
    if os.path.exists(date_range_file):
        os.remove(date_range_file)

    out_csv = open(date_range_file, "w", newline='', encoding='utf-8')
    csv_writer = csv.writer(out_csv)
    csv_header = ['id', 'date_min', 'date_max']
    csv_writer.writerow(csv_header)

    for json_file in tqdm.tqdm(data_files, desc="processing raw dataset",
                               bar_format="%s{l_bar}{bar}{r_bar}%s" % (Fore.BLUE, Fore.RESET)):
        f = open(json_file, 'r', encoding='utf-8')
        f_lines = f.readlines()
        date_max = datetime.datetime(1000, 1, 1)
        date_min = date_max
        for line in f_lines:
            json_data = json.loads(line)
            line_date = datetime.datetime.strptime(json_data['date'] + " " + json_data['time'] + " " +
                                                   json_data['timezone'], "%Y-%m-%d %H:%M:%S %z")
            if date_max == datetime.datetime(1000, 1, 1):
                date_max = line_date
                date_min = date_max
            if line_date > date_max:
                date_max = line_date
            if line_date < date_min:
                date_min = line_date
        csv_writer.writerow([os.path.splitext(os.path.basename(json_file))[0], date_min.strftime("%Y-%m-%d"),
                             date_max.strftime("%Y-%m-%d")])
    
    return date_range_file

        # for item in f:
        #     tweet = json.loads(item)
        #     for line in tweet:
        #     # json_data = json.loads(line)
        #         line_date = datetime.datetime.strptime(line['date'] + " " + line['time'] + " " +
        #                                             line['timezone'], "%Y-%m-%d %H:%M:%S %z")
        #         if date_max == datetime.datetime(1000, 1, 1):
        #             date_max = line_date
        #             date_min = date_max
        #         if line_date > date_max:
        #             date_max = line_date
        #         if line_date < date_min:
        #             date_min = line_date
        # csv_writer.writerow([os.path.splitext(os.path.basename(json_file))[0], date_min.strftime("%Y-%m-%d"),
        #                      date_max.strftime("%Y-%m-%d")])


# if __name__ == '__main__':
#     get_date_range(r'resources\data\raw_data\ww', 'xxxx.csv')
