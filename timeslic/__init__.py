import os
from .get_range import get_date_range
from .segment import get_max_common_range, cut_user_corpus

def time_slicing(coverage, length, raw_dataset_path):
    date_range_path = get_date_range(raw_dataset_path)
    start, end = get_max_common_range(coverage, date_range_path)
    time_sliced_dataset = cut_user_corpus(start, end, raw_dataset_path, length)
    os.remove(date_range_path)
    return time_sliced_dataset
    
