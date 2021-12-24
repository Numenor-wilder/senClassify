class HyperParameter:
    '''超参数类'''
    def __init__(self):
        pass
    

class EmbeddingHyperParma(HyperParameter):
    '''词向量超参数'''
    def __init__(self, dimension, epochs, window):
        self.dimension = dimension
        self.epochs = epochs
        self.window = window
        

class GlobalPath:
    EMBED_MODEL_PATH = r'resources\model\embedding'
    '''词向量模型保存路径'''

    ClASS_MODEL_PATH = r'resources\model\classifier'  
    '''分类器模型保存路径'''

    TFIDF_MODEL_PATH = r'resources\model\tfidf'
    '''TFIDF模型保存路径'''
    
    TFIDF_CORPUS_PATH = r'resources\tfidf_corpus'
    '''.'''
    
    EMBED_PATH =r'resources\embedding'
    '''.'''
    
    TEXT_PATH = r'resources\text'  # 文本
    '''.'''
    
    RAW_DATA_PATH = r'resources\data\raw_data'
    '''原始数据所在路径'''

    LABEL_DATA_PATH = r'resources\data\labeled_data'
    '''标签数据所在路径'''

    DICT_PATH = r'resources\dictionary'
    '''分词所需以及hashtag字典所在路径'''

    CLEAN_DATA_NORMAL = r'resources\clean_data\jieba\segment'
    '''常规分词预处理结果输出路径'''

    CLEAN_DATA_TIMESLICE = r'resources\clean_data\jieba\timeslice'
    '''时间片分词预处理结果输出路径'''

    TMP_DATA_PATH = r'resources\tmp'
    '''临时中间数据保存路径'''

    WK_POINT_PATH = r'resources\text\weakpoint'
    '''.'''

class CustomDict:
    default = 'traditional.txt'
    user = 'user.txt'
    stopword = 'stopwords.txt'
    hashtag = 'hot_hashtags.txt'