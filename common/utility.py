import os
from typing import Union


def check_multi_dir(*multi_dirs: str):
    '''
    当路径不存在时，确定多级文件路径中未创建的具体层级
    :param multi_dirs: 文件路径
    
    :return: args_prefix 已存在的前缀路径, args_dirs 不存在的子路径列表
    '''
    args_prefix = []
    args_dirs = []

    for mlt_dir in multi_dirs:
        dirs = []
        multi_path = mlt_dir

        while(not os.path.exists(multi_path)):
            multi_path, dir = os.path.split(multi_path)
            dirs.append(dir)
            if len(multi_path) == 0:
                break
        
        args_prefix.append(multi_path)
        args_dirs.append(dirs[1:])

    return args_prefix, args_dirs


def create_multi_dir(multi_dir: str, dirs: list):
    '''
    创建多级路径
    '''
    while(len(dirs)):
        dir = dirs[-1]
        dirs.pop()
        multi_dir = os.path.join(multi_dir, dir)
        os.mkdir(multi_dir)


def path_exist_operate(multi_dirs: Union[str, list]):
    multi_dirs, dirs = check_multi_dir(multi_dirs)   # 处理文件路径问题
    for mul_dir, dir in zip(multi_dirs, dirs):
        create_multi_dir(mul_dir, dir)

