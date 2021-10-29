class hyper_parameter:
    def __init__(self, dimension, epochs, window):
        self.dimension = dimension
        self.epochs = epochs
        self.window = window

class path:
    EMBED_MODEL_PATH = r'.\resources\model\embedding'
    ClASS_MODEL_PATH = r'.\resources\model\classifier'
    TFIDF_MODEL_PATH = r'.\resources\model\tfidf'
    CORPUS_PATH = r'.\resources\corpus'
    EMBED_PATH =r'.\resources\embedding'
    DATA_PATH = r'.\data' # 训练数据
    TEXT_PATH = r'.\resources\text'  # 文本
    RAW_PATH = r'.\resources\raw_data'  # 原始数据