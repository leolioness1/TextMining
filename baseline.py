# Import the necessary modules
import unicodedata
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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from ptbr_postag.ptbr_postag import pos_tagger

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
# for windows10 run: pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import pickle
import warnings
import datetime
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


def split_strings_n_words(df, n):
    new_df = pd.concat([pd.Series(row['author'], [' '.join(row['text'].split()[x:x + n]) for x in range(0, len(row['text'].split()), n)]) for _, row in df.iterrows()]).reset_index()
    # new data frame with split value columns
    new = new_df[0].str.split("/", n=1, expand=True)
    new_df.drop(columns=[0], inplace=True)
    new_df["author"] = new[0]
    new_df["title"] = new[1]
    new_df.rename(columns={"index":"text"}, inplace=True)
    return new_df

new_file_data = split_strings_n_words(file_data,500)
new_file_data_1000 = split_strings_n_words(file_data,1000)

new_file_data.groupby(['author','title']).count().to_csv("number_of_500_samples_per_title_per_author.csv")

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
        # nfkd_form = unicodedata.normalize('NFKD', text)
        # text = nfkd_form.encode('ascii', 'ignore').decode()
        #REMOVE TAGS
        text = BeautifulSoup(text).get_text()
        processed_corpus.append(text)
    return processed_corpus

cleaned_documents= clean_data(new_file_data)
cleaned_documents_1000= clean_data(new_file_data_1000)
cleaned_documents_test=clean_data(file_data_test)

def update_df(dataframe, cleaned_documents):
    dataframe['text'] = cleaned_documents

update_df(new_file_data, cleaned_documents)
update_df(new_file_data, cleaned_documents_1000)
update_df(file_data_test,cleaned_documents_test)

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

###stem_file_data and stem_documents also has stemming and stopwords removed##
#to be used for NaiveBayes etc retains less text info##

stem_documents = stem_stop_words(new_file_data)
stem_documents_test = stem_stop_words(file_data_test)

stem_file_data = new_file_data.copy(deep=True)
stem_file_data_test = file_data_test.copy(deep=True)
stem_file_data['text'] = stem_documents
stem_file_data_test['text'] = stem_documents_test


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

###lemma_file_data and lemma_documents also has lemmatisation and stopwords removed##
#to be used for NaiveBayes etc retains less text info##

lemma_documents = lemma_stop_words(new_file_data)
lemma_documents_test = lemma_stop_words(file_data_test)

lemma_file_data = new_file_data.copy(deep=True)
lemma_file_data_test = file_data_test.copy(deep=True)
lemma_file_data['text'] = lemma_documents
lemma_file_data_test['text'] = lemma_documents_test


############ word count #################

word_count = file_data['text'].apply(lambda x: len(str(x).split(" ")))
file_data['word_count']=word_count
file_data[['text','word_count']].head()

all_words = ' '.join(file_data['text']).split()
freq = pd.Series(all_words).value_counts()
freq[:25]

tagger = pos_tagger()

file_data_pos = []
for i in file_data.text:
    file_data_pos.append(tagger.tag_text(text = i,bRemoveStopwords=False))
postagsfull = []

for text in file_data_pos:
    postext = ""
    for token in text:
        postext = postext+token[1]+" "
    postagsfull.append(postext)
cv = CountVectorizer(max_df = 0.8,ngram_range=(1,3))
vector_count = cv.fit_transform(postagsfull)

tfidf_vectorizer = TfidfTransformer()
vector_tfidf = tfidf_vectorizer.fit_transform(vector_count)


cv = CountVectorizer(
    max_df=0.8,
    max_features=10000,
    ngram_range=(1,3), # only bigram (2,2)
    strip_accents = 'ascii'
)


X = cv.fit_transform(cleaned_documents)

list(cv.vocabulary_.keys())[:10]



############# BAG OF WORDS ##################
cv_NB = CountVectorizer(max_df=0.9, binary=True)

X_NB = cv_NB.fit_transform(stem_file_data['text'])


cv_KNN = CountVectorizer(
    max_df=0.8,
    max_features=1000,
    ngram_range=(1,3) # only bigram (2,2)
)

X_KNN = cv_KNN.fit_transform(stem_file_data['text'])


y = np.array(stem_file_data["author"])


################## KNN ######################

X_train, X_test, y_train, y_test = train_test_split(X_KNN, y, test_size = 0.2, random_state = 42)

modelknn = KNeighborsClassifier(n_neighbors=10, weights='distance', algorithm='brute', leaf_size=30, p=2,
                                          metric='cosine', metric_params=None, n_jobs=1)
modelknn.fit(X_train,y_train)
modelknn.classes_


predKNN = modelknn.predict(X_test)
predKNN
print (classification_report(predKNN, y_test))

################ Naive Bayes #################

X_train, X_test, y_train, y_test = train_test_split(X_NB, y, test_size = 0.1, random_state = 42)

nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(X_train,y_train)

# Create the predicted tags: pred
predNB = nb_classifier.predict(X_test)
predNB

print(classification_report(predNB, y_test))

#EMBEDDINGS

def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens

tokenized_corpus = tokenize_corpus(cleaned_documents)
vocabulary = {word for doc in tokenized_corpus for word in doc}
vocab_size = len(vocabulary)


# The maximum number of words to be used. (most frequent) or could use whole vocab size
MAX_NB_WORDS = 60000

epochs =8
# “batch gradient descent“ batch_size= len(X_train) epochs=200
batch_size = 32

tokenizer = Tokenizer()
tokenizer_1000= Tokenizer()
tokenizer.fit_on_texts(new_file_data.text.values)
tokenizer_1000.fit_on_texts(new_file_data_1000.text.values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# Change text for numerical ids and pad
X = tokenizer.texts_to_sequences(new_file_data.text)
X_1000 = tokenizer_1000.texts_to_sequences((new_file_data_1000.text))
X_1000 = pad_sequences(X_1000,maxlen=1000)
X = pad_sequences(X, maxlen=500)
print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(new_file_data['author'])
Y_1000 = pd.get_dummies(new_file_data_1000['author'])
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42, stratify=Y)
X_train_1000, X_test_1000, y_train_1000, y_test_1000 = train_test_split(X_1000,Y_1000, test_size=0.2, random_state=42,stratify=Y_1000)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

model = Sequential()
model.add(Embedding(input_dim=MAX_NB_WORDS, output_dim=100, input_length=X.shape[1]))
model.add(LSTM(100))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_1000 = Sequential()
model_1000.add(Embedding(input_dim=MAX_NB_WORDS, output_dim=100, input_length=X.shape[1]))
model_1000.add(LSTM(100))
model_1000.add(Dense(6, activation='softmax'))
model_1000.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,callbacks=[EarlyStopping(monitor='loss', patience=3, min_delta=0.01)])
history_1000 = model_1000.fit(X_train_1000, y_train_1000, epochs=epochs, batch_size=batch_size,callbacks=[EarlyStopping(monitor='loss', patience=3, min_delta=0.01)])

# save the model to disk
filename = 'lstm_model_{}.pkl'.format(datetime.datetime.today().strftime("%d_%m_%Y_%H_%M_%S"))
filename_1000 = 'lstm_model_1000_{}.pkl'.format(datetime.datetime.today().strftime("%d-%m-%Y_%H_%M_%S"))
pickle.dump(model, open(filename, 'wb'))
pickle.dump(model_1000, open(filename_1000, 'wb'))

#
# # load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))

# evaluate the model
train_acc = model.evaluate(X_train, y_train, verbose=0)
test_acc = model.evaluate(X_test, y_test, verbose=0)
print('---------------------------------------500 word samples------------------------------')
print('Train set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(train_acc[0],train_acc[1]))
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(test_acc[0],test_acc[1]))

# evaluate the model
train_acc_1000 = model_1000.evaluate(X_train_1000, y_train_1000, verbose=0)
test_acc_1000 = model_1000.evaluate(X_test_1000, y_test_1000, verbose=0)
print('---------------------------------------1000 word samples------------------------------')
print('Train set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(train_acc_1000[0],train_acc_1000[1]))
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(test_acc_1000[0],test_acc_1000[1]))


# plot training history
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.title("500 sample")
plt.show()

# plot training history
plt.plot(history_1000.history['accuracy'], label='train')
plt.plot(history_1000.history['val_accuracy'], label='test')
plt.legend()
plt.title("500 sample")
plt.show()

# Change text for numerical ids and pad
X_new = tokenizer.texts_to_sequences(file_data_test.text)
X_new = pad_sequences(X_new, maxlen=1000)


# Use the model to predict on new data
predicted = model.predict(X_new)
predicted_1000= model_1000.predict(X_new)
# Choose the class with higher probability
file_data_test.insert(1,'prediction_6',Y.columns[list(np.argmax(predicted, axis=1))])
file_data_test.insert(1,'prediction_1000_2',Y.columns[list(np.argmax(predicted_1000, axis=1))])

# # Compute and print the confusion matrix
# print(confusion_matrix(y_true, y_pred))
#
# # Create the performance report
# print(classification_report(y_true, y_pred, target_names=news_cat))


# # Initialize a CountVectorizer object: count_vectorizer
# count_vectorizer = CountVectorizer()
#
# # Transform the training data using only the 'text' column values: count_train
# count_train = count_vectorizer.fit_transform(X_train)
#
# # Transform the test data using only the 'text' column values: count_test
# count_test = count_vectorizer.transform(X_test)
#
# # Print the first 10 features of the count_vectorizer
# print(count_vectorizer.get_feature_names()[:10])
#
#
# # Initialize a TfidfVectorizer object: tfidf_vectorizer
# tfidf_vectorizer = TfidfVectorizer( max_df=0.7)
#
# # Transform the training data: tfidf_train
# tfidf_train = tfidf_vectorizer.fit_transform(X_train)
#
# # Transform the test data: tfidf_test
# tfidf_test = tfidf_vectorizer.transform(X_test)
#
# # Print the first 10 features
# print(tfidf_vectorizer.get_feature_names()[:10])
#
# # Print the first 5 vectors of the tfidf training data
# print(tfidf_train.A[:5])
#
# # Create the CountVectorizer DataFrame: count_df
# count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())
#
# # Create the TfidfVectorizer DataFrame: tfidf_df
# tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())
#
# # Print the head of count_df
# print(count_df.head())
#
# # Print the head of tfidf_df
# print(tfidf_df.head())
#
# # Calculate the difference in columns: difference
# difference = set(count_df.columns) - set(tfidf_df.columns)
# print(difference)
#
# # Check whether the DataFrames are equal
# print(count_df.equals(tfidf_df))
#
#
# # Instantiate a Multinomial Naive Bayes classifier: nb_classifier
# nb_classifier = MultinomialNB()
#
# # Fit the classifier to the training data
# nb_classifier.fit(count_train,y_train)
#
# # Create the predicted tags: pred
# pred = nb_classifier.predict(count_test)
#
# # Calculate the accuracy score: score
# score = metrics.accuracy_score(y_test,pred)
# print(score)
#
# # Calculate the confusion matrix: cm
# cm = metrics.confusion_matrix(y_test,pred)
# print(cm)
#
# # Create the list of alphas: alphas
# alphas = np.arange(0,1,step=0.1)
#
# # Define train_and_predict()
# def train_and_predict(alpha):
#     # Instantiate the classifier: nb_classifier
#     nb_classifier = MultinomialNB(alpha=alpha)
#     # Fit to the training data
#     nb_classifier.fit(tfidf_train, y_train)
#     # Predict the labels: pred
#     pred = nb_classifier.predict(tfidf_test)
#     # Compute accuracy: score
#     score = metrics.accuracy_score(y_test,pred)
#     return score
#
# # Iterate over the alphas and print the corresponding score
# for alpha in alphas:
#     print('Alpha: ', alpha)
#     print('Score: ', train_and_predict(alpha))
#
#
# # Get the class labels: class_labels
# class_labels = nb_classifier.classes_
#
# # Extract the features: feature_names
# feature_names = tfidf_vectorizer.get_feature_names()
#
# # Zip the feature names together with the coefficient array and sort by weights: feat_with_weights
# feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))
#
# # Print the first class label and the top 20 feat_with_weights entries
# print(class_labels[0], feat_with_weights[:20])
#
# # Print the second class label and the bottom 20 feat_with_weights entries
# print(class_labels[1], feat_with_weights[-20:])
