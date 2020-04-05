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
from sklearn.model_selection import StratifiedKFold
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
new_file_data_1000.groupby(['author','title']).count().to_csv("number_of_1000_samples_per_title_per_author.csv")

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
update_df(new_file_data_1000, cleaned_documents_1000)
update_df(file_data_test,cleaned_documents_test)

file_data_test.insert(1,'y_true',['JoseSaramago','AlmadaNegreiros','LuisaMarquesSilva','EcaDeQueiros','CamiloCasteloBranco','JoseRodriguesSantos','JoseSaramago','AlmadaNegreiros','LuisaMarquesSilva','EcaDeQueiros','CamiloCasteloBranco','JoseRodriguesSantos'])


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

# test = ['pg22801.txt','pg26103.txt','pg25641.txt','Furia Divina - Jose Rodrigues dos Santos.txt',
#        'O Homem Duplicado - Jose Saramago.txt','UltimaHistoria.txt']pg22801.txt
test=['pg22801.txt','pg27364.txt','pg25641.txt','O Setimo Selo - Jose Rodrigues dos Santos.txt','O Homem Duplicado - Jose Saramago.txt','ABelaHistoria.txt']

new_file_data_subset = new_file_data.loc[~new_file_data.title.isin(test)]
new_file_data_test = new_file_data.loc[new_file_data.title.isin(test)]

y_train = np.array(new_file_data['author'].loc[~new_file_data.title.isin(test)])
y_test = np.array(new_file_data['author'].loc[new_file_data.title.isin(test)])



################## KNN ######################

def KNN(train_df,test_df):
    cv_KNN = CountVectorizer(
        max_df=0.8,
        max_features=1000,
        ngram_range=(1,3) # only bigram (2,2)
    )
    X_KNN = cv_KNN.fit_transform(train_df['text'])

    y = np.array(train_df["author"])

    X_train, X_test, y_train, y_test = train_test_split(X_KNN, y, test_size = 0.2, random_state = 42)

    modelknn = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='brute', leaf_size=30, p=2,
                                              metric='cosine', metric_params=None, n_jobs=1)
    modelknn.fit(X_train, y_train)

    X_KNN_new = cv_KNN.fit_transform(test_df['text'])
    predKNN = modelknn.predict(X_test)
    print(classification_report(y_test,predKNN))
    predKNN_new = modelknn.predict(X_KNN_new)
    file_data_test["pred_KNN"] = predKNN_new
    print(classification_report(file_data_test['y_true'],predKNN_new))
    return predKNN_new

KNN(lemma_file_data,lemma_file_data_test)

# ################ Naive Bayes #################
# def NB(train_df,test_df):
#     cv_NB = CountVectorizer(max_df=0.9)
#     X_NB = cv_NB.fit_transform(train_df['text'])
#     y = np.array(train_df["author"])
#     X_train, X_test, y_train, y_test = train_test_split(X_NB, y, test_size = 0.2, random_state = 42)
#
#     nb_classifier = MultinomialNB()
#     # Fit the classifier to the training data
#     nb_classifier.fit(X_train,y_train)
#     X_NB_new = cv_NB.fit_transform(test_df['text'])
#     # Create the predicted tags: pred
#     predNB = nb_classifier.predict(X_test)
#     print(classification_report(y_test,predNB))
#     predNB_new = nb_classifier.predict(X_NB_new)
#     file_data_test["pred_NB"] = predNB_new
#     print(classification_report(file_data_test["y_true"],predNB_new))
#     return predNB_new
#
# NB(lemma_file_data,lemma_file_data_test)


#EMBEDDINGS
def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens

tokenized_corpus = tokenize_corpus(cleaned_documents)
vocabulary = {word for doc in tokenized_corpus for word in doc}
vocab_size = len(vocabulary)
print(vocab_size)

# The maximum number of words to be used. (most frequent) or could use whole vocab size
def LSTM_model(train_df,test_df,new_df,MAX_LEN,MAX_NB_WORDS,epochs,batch_size):

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
    tokenizer.fit_on_texts(train_df.text.values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # Change text for numerical ids and pad
    X = tokenizer.texts_to_sequences(train_df.text)
    X = pad_sequences(X, maxlen=MAX_LEN)
    print('Shape of data tensor:', X.shape)
    Y = pd.get_dummies(train_df['author'])
    X_train, X_dev, y_train, y_dev = train_test_split(X, Y, test_size=0.1, random_state=7, stratify=Y)
    print(X_train.shape, y_train.shape)
    print(X_dev.shape, y_dev.shape)
    # kfold_splits=10
    # # Instantiate the cross validator
    # skf = StratifiedKFold(n_splits=kfold_splits, shuffle=True,random_state=7)
    # # Loop through the indices the split() method returns
    # for index, (train_indices, val_indices) in enumerate(skf.split(X, Y.values.argmax(1))):
    #     print
    #     "Training on fold " + str(index + 1) + "/10..."
    #     # Generate batches from indices
    #     X_train, X_dev = X[train_indices], X[val_indices]
    #     y_train, y_dev = Y[train_indices], Y[val_indices]
        # Clear model, and create it
    model = Sequential()
    model.add(Embedding(input_dim=MAX_NB_WORDS, output_dim=100, input_length=X.shape[1]))
    model.add(LSTM(100))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Debug message I guess
    # print "Training new iteration on " + str(xtrain.shape[0]) + " training samples, " + str(xval.shape[0]) + " validation samples, this may be a while..."

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_data=(X_dev,y_dev),callbacks=[EarlyStopping(monitor='loss', patience=3, min_delta=0.01)])
    # accuracy_history = history.history['acc']
    # val_accuracy_history = history.history['val_acc']
    # print( "Last training accuracy: " + str(accuracy_history[-1]) + ", last validation accuracy: " + str(val_accuracy_history[-1]))

    # save the model to disk
    filename = 'lstm_model_{}_{}.pkl'.format(MAX_LEN,datetime.datetime.today().strftime("%d_%m_%Y_%H_%M_%S"))
    pickle.dump(model, open(filename, 'wb'))
    # # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))
    # evaluate the model
    train_acc = model.evaluate(X_train, y_train, verbose=0)
    # Change text for numerical ids and pad
    X_test = tokenizer.texts_to_sequences(test_df.text)
    X_test = pad_sequences(X_test, maxlen=MAX_LEN)
    test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('---------------------------------------{} word samples------------------------------'.format(MAX_LEN))
    print('Train set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(train_acc[0],train_acc[1]))
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(test_acc[0],test_acc[1]))

    # plot training history
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.title("{} sample".format(MAX_LEN))
    plt.show()

    # Change text for numerical ids and pad
    X_new = tokenizer.texts_to_sequences(new_df.text)
    X_new = pad_sequences(X_new, maxlen=MAX_LEN)

    # Use the model to predict on new data
    predicted = model.predict(X_new)

    # Choose the class with higher probability
    test_df.drop('prediction',inplace=True)
    test_df.insert(2,'prediction',Y.columns[list(np.argmax(predicted, axis=1))])

    # Create the performance report
    print(classification_report(file_data_test['y_true'], predicted, target_names=Y.columns))
    return test_df.head()

LSTM_model(new_file_data_subset, new_file_data_test, file_data_test,1000,90000,5,32)