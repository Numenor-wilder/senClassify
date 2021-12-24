import csv
import os

def export_result_to_csv_timespan_and_attitude_from_csvfile(input_file, output_file, threshold=0.5):
    '''
    export_result_to_csv_timespan_and_attitu.de_from_csvfile
    :param input_file: 输入文件路径，例如：data/user_twoClasses_label_probability_week1.csv
    :param output_file: 输出文件路径，例如：output/user_change_timespan1.csv
    :param threshold: 设置的阈值，默认0.5
    :return:
    '''
    input_open_file = open(input_file, 'r', encoding='utf-8')
    output_open_file = open(output_file, 'w', encoding='utf-8', newline='')
    reader = csv.DictReader(input_open_file)
    writer = csv.writer(output_open_file)
    writer.writerow(['user_id', 'pre_timespan', 'cur_timespan', 'pre_proba', 'cur_proba'])

    for line_dict in reader:
        id = line_dict['NO']
        for i in range(2, len(line_dict)):
            pre = float(line_dict[str(i)])
            cur = float(line_dict[str(i - 1)])
            if abs(pre - cur) > threshold and pre != 0 and cur != 0:
                print("pre_timespan =", i - 1, "cur_timespan =", i, "pre_prob =", pre, "cur_prob =", cur, "id =", id, )
                writer.writerow([id, i - 1, i, pre, cur])
    input_open_file.close()
    output_open_file.close()


# if __name__ == '__main__':
#     proba_dir = r'resources\text\weakpoint\probability'
#     file_list = os.listdir(proba_dir)

#     for file in file_list:
#         input_file = os.path.join(proba_dir, file)
#         output_file = fr'resources/changing/{file}'
#         export_result_to_csv_timespan_and_attitude_from_csvfile(input_file, output_file)


