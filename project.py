# Import the necessary modules
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer
from bs4 import BeautifulSoup
import string
from gensim.corpora.dictionary import Dictionary
from collections import defaultdict
import itertools
import nltk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from gensim.models.tfidfmodel import TfidfModel
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
# run this if not installed: python -m spacy download pt_core_news_sm
import pt_core_news_sm
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

root = os.getcwd() + '\\Corpora\\train'

file_name_and_text={}
for file in os.listdir(root):
    name = os.path.splitext(file)[0]
    print('{}/{}'.format(root,name))

    for file in os.listdir('{}/{}'.format(root,name)):
        if file.endswith(".txt"):
            try:
                with open(os.path.join(root, name,file), "r", encoding='utf-8') as target_file:
                    file_name_and_text['{}/{}'.format(name,file)] = target_file.read()
            except Exception as e:
                print("{} generated an error: \n {}".format(os.path.join(root, name,file)),e)

file_data = (pd.DataFrame.from_dict(file_name_and_text, orient='index')
             .reset_index().rename(index = str, columns = {'index': 'author', 0: 'text'}))
# new data frame with split value columns
new = file_data["author"].str.split("/", n=1, expand=True)
file_data["author"] = new[0]
file_data["Title"] = new[1]

root_test = os.getcwd() + '\\Corpora\\test'

file_name_and_text_test={}
for file in os.listdir(root_test):
    name = os.path.splitext(file)[0]
    print('{}/{}'.format(root_test,name))

    for file in os.listdir('{}/{}'.format(root_test,name)):
        if file.endswith(".txt"):
            try:
                with open(os.path.join(root_test, name,file), "r", encoding='utf-8') as target_file:
                    file_name_and_text_test['{}/{}'.format(name,file)] = target_file.read()
            except Exception as e:
                print("{} generated an error: \n {}".format(os.path.join(root_test, name,file)),e)

file_data_test = (pd.DataFrame.from_dict(file_name_and_text_test, orient='index')
             .reset_index().rename(index = str, columns = {'index': 'number_of_words', 0: 'text'}))



stop = set(stopwords.words('portuguese'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
stemmer = SnowballStemmer('portuguese')


def clean_data(dataframe):
    """
    Function that a receives a list of strings and preprocesses it.
    :param text_list: List of strings.
    :param lemmatize: Tag to apply lemmatization if True.
    :param stemmer: Tag to apply the stemmer if True.
    """
    processed_corpus = []
    for i in range(len(dataframe)):
        text = dataframe['text'][i]
        #LOWERCASE TEXT
        text = text.lower()
        #REMOVE NUMERICAL DATA AND PUNCTUATION
        text = re.sub("[^a-zA-Z-ÁÀÂÃâáàãçÉÈÊéèêúùÚÙÕÓÒÔôõóòÍÌíìçÇ]", ' ', text)
        #REMOVE TAGS
        text = BeautifulSoup(text).get_text()
        processed_corpus.append(text)
    return processed_corpus

cleaned_documents= clean_data(file_data)
cleaned_documents_test=clean_data(file_data_test)

def update_df(dataframe, cleaned_documents):
    dataframe['text'] = cleaned_documents

update_df(file_data, cleaned_documents)
update_df(file_data_test, cleaned_documents_test)

###file_data nd clean documents only has non-alpha characters and html removed##
#to be used for language modelling retains most text info##

def stem_stop_words (dataframe):
    processed_corpus = []
    for i in range(len(dataframe)):
        text = dataframe['text'][i]
        # REMOVE STOP WORDS
        text = text.split()
        text = [stemmer.stem(word) for word in text if not word in stop]
        text = " ".join(text)
        processed_corpus.append(text)
    return processed_corpus

def lemma_stop_words (dataframe):
    processed_corpus = []
    for i in range(len(dataframe)):
        text = dataframe['text'][i]
        # REMOVE STOP WORDS
        text = text.split()
        text = [lemma.lemmatize(word) for word in text if not word in stop]
        text = " ".join(text)
        processed_corpus.append(text)
    return processed_corpus

###lem_file_data and lem_documents also has lemmatisation and stopwords removed##
#to be used for NaiveBayes etc retains less text info##

stem_documents = stem_stop_words(file_data)
stem_documents_test = stem_stop_words(file_data_test)

stem_file_data = file_data.copy(deep=True)
stem_file_data_test = file_data_test.copy(deep=True)
stem_file_data['text'] = stem_documents
stem_file_data_test['text'] = stem_documents_test


lemma_documents = lemma_stop_words(file_data)
lemma_documents_test = lemma_stop_words(file_data_test)

lemma_file_data = file_data.copy(deep=True)
lemma_file_data_test = file_data_test.copy(deep=True)
lemma_file_data['text'] = lemma_documents
lemma_file_data_test['text'] = lemma_documents_test

############ word count #################

word_count  = file_data['text'].apply(lambda x: len(str(x).split(" ")))
file_data['word_count']=word_count
file_data[['text','word_count']].head()

all_words = ' '.join(file_data['text']).split()
freq = pd.Series(all_words).value_counts()
freq[:25]


########## n grams ##################

def get_top_n_grams(corpus, top_k, n):

    vec = CountVectorizer(ngram_range=(n, n), max_features=2000).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0)

    words_freq = []
    for word, idx in vec.vocabulary_.items():
        words_freq.append((word, sum_words[0, idx]))

    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    top_df = pd.DataFrame(words_freq[:top_k])
    top_df.columns = ["Ngram", "Freq"]

    return top_df

def plot_frequencies(top_df):
    
    x_labels = top_df["Ngram"][:30]
    y_pos = np.arange(len(x_labels))
    values = top_df["Freq"][:30]
    plt.bar(y_pos, values, align='center', alpha=0.5)
    plt.xticks(y_pos, x_labels)
    plt.ylabel('Frequencies')
    plt.title('Words')
    plt.xticks(rotation=90)
    plt.show()

top_df2 = get_top_n_grams(lemma_documents, top_k=10, n=2)
top_df3 = get_top_n_grams(lemma_documents, top_k=10, n=3)

top_df2
top_df3

plot_frequencies(top_df2)
plot_frequencies(top_df3)


########### TFIDF ############

cv = CountVectorizer(
    max_df=0.8,
    max_features=1000,
    ngram_range=(1,3) # only bigram (2,2)
)

X = cv.fit_transform(lemma_documents)
X_new = cv.fit_transform(lemma_documents)

list(cv.vocabulary_.keys())[:10]

tfidf_vectorizer = TfidfTransformer()
tfidf_vectorizer.fit(X)

# get feature names
feature_names = cv.get_feature_names()

# fetch document for which keywords needs to be extracted
doc = lemma_documents[10]  # 532

# generate tf-idf for the given document
tf_idf_vector = tfidf_vectorizer.transform(cv.transform([doc]))

tf_idf_vector.toarray()

def extract_feature_scores(feature_names, document_vector):
    
    feature2score = {}
    for i in range(len(feature_names)):
        feature2score[feature_names[i]] = document_vector[0][i]
    return sorted(feature2score.items(), key=lambda kv: kv[1], reverse=True)

extract_feature_scores(feature_names, tf_idf_vector.toarray())[:10]



############ BAG OF WORDS #################

cv = CountVectorizer(max_df=0.9, binary=True)
X = cv.fit_transform(lemma_file_data['text'])
y = np.array(lemma_file_data["author"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

################## KNN ######################

modelknn = KNeighborsClassifier(n_neighbors=10, weights='distance', algorithm='brute', leaf_size=30, p=2,
                                         metric='cosine', metric_params=None, n_jobs=1)
modelknn.fit(X_train,y_train)
modelknn.classes_
y_train.shape
y_test.shape
X_test.shape

predKNN = modelknn.predict(X_test)
predKNN
print (classification_report(predKNN, y_test))

################ Naive Bayes #################

# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(X_train,y_train)

# Create the predicted tags: pred
predNB = nb_classifier.predict(X_test)
predNB
print (classification_report(predNB, y_test))


########################## LINEAR CLASSIFIERS #################################

class Classifier(object):
    
    def __init__(self, input_size, n_classes):
        
        self.parameters = np.zeros((input_size + 1, n_classes))  # input_size +1 to include the Bias term

    def train(self, X, Y, devX, devY, epochs=20):
        
        train_accuracy = [self.evaluate(X, Y)]
        dev_accuracy = [self.evaluate(devX, devY)]
        for epoch in range(epochs):
            for i in tqdm(range(X.shape[0])):
                self.update_weights(X[i, :], Y[i])
            train_accuracy.append(self.evaluate(X, Y))
            dev_accuracy.append(self.evaluate(devX, devY))
        return train_accuracy, dev_accuracy

    def evaluate(self, X, Y):
        
        correct_predictions = 0
        for i in range(X.shape[0]):
            y_pred = self.predict(X[i, :])
            if Y[i] == y_pred:
                correct_predictions += 1
        return correct_predictions / X.shape[0]

    def plot_train(self, train_accuracy, dev_accuracy):
       
        x_axis = [epoch + 1 for epoch in range(len(train_accuracy))]
        plt.plot(x_axis, train_accuracy, '-g', linewidth=1, label='Train')
        plt.xlabel("epochs")
        plt.ylabel("Accuracy")
        plt.plot(x_axis, dev_accuracy, 'b-', linewidth=1, label='Dev')
        plt.legend()
        plt.show()

    def update_weights(self, x, y):
        
        pass

    def predict(self, x):
        
        pass


class MultinomialLR(Classifier):

    def __init__(self, input_size, n_classes, lr=0.001):
        
        Classifier.__init__(self, input_size, n_classes)
        self.lr = lr

    def predict(self, input):
        
        return np.argmax(self.softmax(np.dot(np.append(input, [1]), self.parameters)))

    def softmax(self, x):
        
        return np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)), axis=0)

    def update_weights(self, x, y):
        
        linear = np.dot(np.append(x, [1]), self.parameters)
        predictions = self.softmax(linear)
        self.parameters = self.parameters - self.lr*(np.outer(predictions, np.append(x, [1])).T)
        self.parameters[:, y] = self.parameters[:, y] + self.lr*np.append(x, [1])
