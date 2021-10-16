from classifier import Classifier

def main():
    embed_path = 'fasttext_embedding.csv'
    classifier = Classifier(epath=embed_path)
    features, labels = classifier.get_training_data(classifier.epath)
    classifier.svm_training(features, labels)
    

if __name__ == '__main__':
    main()