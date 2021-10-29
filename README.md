# senClassify

情感分类


# 资源文件

├─corpus  语料库保存
│  └─dic  字典保存
├─embedding  embedding保存  
│  └─fasttext       
├─model
│  ├─classifier  分类器模型保存  
│  ├─embedding  词向量模型保存  
│  │  ├─doc2vec     
│  │  └─fasttext    
│  └─tfidf  TF-IDF模型
├─raw_data
│  └─ww   原始推文数据  
└─text
    ├─fenci
    │  ├─dict  分词字典  
    │  └─word_seg  分词结果  
    │      ├─debug
    │      ├─test
    │      └─ww_202110
    └─tweets_bert


├─resources  
│  ├─embedding
│  │  └─fasttext  
│  ├─model  词向量模型保存路径  
│  │  ├─fasttext  
│  │  └─word2vec  
│  ├─raw_data  
│  │  └─ww  原始推文数据   
│  └─text  
│      ├─fenci  
│      │  ├─dict 分词字典  
│      │  ├─tweets 单个用户推文时间线  
│      │  └─word_seg 分词结果  
│      └─tweets_bert  


# 训练数据

data 训练标签数据  


# 依赖

dependency:  
python 3.6+  
sklearn  
numpy  
pandas  
gensim  
jieba  
