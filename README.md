# senClassify

情感分类


# 资源文件

raw_data\ww 原始推文数据  
text 推文文本相关  
|--fenci  
   |--dict 分词字典  
   |--tweets 单个用户推文时间线  
   |--word_seg 分词结果  
|--tweets_bert  
model 词向量模型保存路径  
embedding embedding保存路径   


# 训练数据

data 训练数据，包括训练数据和标签数据  


# 依赖

dependency:  
python 3.6+  
sklearn  
numpy  
pandas  
gensim  
jieba  
