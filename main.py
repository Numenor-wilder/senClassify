from datetime import datetime
import logging

from classifier import Classifier
from embedding import Text_embedding
from parameter import hyper_parameter as hparam
from tfidf import Tfidf

label_path = 'new_all_features.csv'
feature_path = 'ww_202110'
doc_path = r'fenci\word_seg\ww_202110'

def main():
    classifier = Classifier()
    features, labels = get_training_data(classifier, feature_path, label_path)
    classifier.svm_training(features, labels)


def get_training_data(classifier, feature_path, label_path):    # 获取训练数据，返回特征和标签向量
    features, labels = classifier.load_training_data(feature_path, label_path)

    if features is None:
        features = get_embedding(doc_path)
        merge = features.merge(labels, how='inner', on='NO')
        features = features.loc[features['NO'].isin(merge['NO'])].iloc[:,1:]
        labels = labels.loc[labels['NO'].isin(merge['NO'])].iloc[:,1:]

    return features, labels


def get_embedding(doc_path):    # 获取文本embedding
    text_embed = Text_embedding(vocab=doc_path)
    tfidf = Tfidf(doc_path)
    if tfidf.dct is None:
        tfidf.dct = tfidf.make_dict(doc_path, tfidf.dataset)
    if tfidf.corpus is None:
        tfidf.corpus = tfidf.make_corpus(tfidf.dataset, tfidf.dct)
    if tfidf.model is None:
        tfidf.model = tfidf.train_tfidf_model(doc_path)

    if text_embed.model is None:
        text_embed.embedding_train()
    embedding = text_embed.get_embedding(doc=doc_path, tfidf=tfidf)

    return embedding


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO)
    main()