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

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
# for windows10 run: pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
from torch.autograd import Variable
import torch.nn.functional as F
import torch

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
        # nfkd_form = unicodedata.normalize('NFKD', text)
        # text = nfkd_form.encode('ascii', 'ignore').decode()
        #REMOVE TAGS
        text = BeautifulSoup(text).get_text()
        processed_corpus.append(text)
    return processed_corpus

cleaned_documents= clean_data(file_data)
cleaned_documents_test=clean_data(file_data_test)

def update_df(dataframe, cleaned_documents):
    dataframe['text'] = cleaned_documents

update_df(file_data, cleaned_documents)
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

stem_documents = stem_stop_words(file_data)
stem_documents_test = stem_stop_words(file_data_test)

stem_file_data = file_data.copy(deep=True)
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

lemma_documents = lemma_stop_words(file_data)
lemma_documents_test = lemma_stop_words(file_data_test)

lemma_file_data = file_data.copy(deep=True)
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


cv = CountVectorizer(
    max_df=0.8,
    max_features=10000,
    ngram_range=(1,3), # only bigram (2,2)
    strip_accents = 'ascii'
)


X = cv.fit_transform(cleaned_documents)

list(cv.vocabulary_.keys())[:10]

def plot_frequencies(top_df):
    """
    Function that receives a dataframe from the "get_top_n_grams" function
    and plots the frequencies in a bar plot.
    """
    x_labels = top_df["Ngram"][:30]
    y_pos = np.arange(len(x_labels))
    values = top_df["Freq"][:30]
    plt.bar(y_pos, values, align='center', alpha=0.5)
    plt.xticks(y_pos, x_labels)
    plt.ylabel('Frequencies')
    plt.title('Words')
    plt.xticks(rotation=90)
    plt.show()


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

top_df = get_top_n_grams(cleaned_documents, top_k=20, n=2)

top_df.head(10)

plot_frequencies(top_df)


# #EMBEDDINGS
#
# def tokenize_corpus(corpus):
#     tokens = [x.split() for x in corpus]
#     return tokens
#
# tokenized_corpus = tokenize_corpus(cleaned_documents)
# vocabulary = {word for doc in tokenized_corpus for word in doc}
# word2idx = {w:idx for (idx, w) in enumerate(vocabulary)}
#
# def build_training(tokenized_corpus, word2idx, window_size=2):
#     window_size = 2
#     idx_pairs = []
#
#     # for each sentence
#     for sentence in tokenized_corpus:
#         indices = [word2idx[word] for word in sentence]
#         # for each word, threated as center word
#         for center_word_pos in range(len(indices)):
#             # for each window position
#             for w in range(-window_size, window_size + 1):
#                 context_word_pos = center_word_pos + w
#                 # make soure not jump out sentence
#                 if context_word_pos < 0 or \
#                         context_word_pos >= len(indices) or \
#                         center_word_pos == context_word_pos:
#                     continue
#                 context_word_idx = indices[context_word_pos]
#                 idx_pairs.append((indices[center_word_pos], context_word_idx))
#     return np.array(idx_pairs)
#
#
# training_pairs = build_training(tokenized_corpus, word2idx)
#
#
# def get_onehot_vector(word_idx, vocabulary):
#     x = torch.zeros(len(vocabulary)).float()
#     x[word_idx] = 1.0
#     return x
#
#
# def Skip_Gram(training_pairs, vocabulary, embedding_dims=5, learning_rate=0.001, epochs=10):
#     torch.manual_seed(3)
#     W1 = Variable(torch.randn(embedding_dims, len(vocabulary)).float(), requires_grad=True)
#     W2 = Variable(torch.randn(len(vocabulary), embedding_dims).float(), requires_grad=True)
#     losses = []
#     for epo in range(epochs):
#         loss_val = 0
#         for input_word, target in training_pairs:
#             x = Variable(get_onehot_vector(input_word, vocabulary)).float()
#             y_true = Variable(torch.from_numpy(np.array([target])).long())
#
#             # Matrix multiplication to obtain the input word embedding
#             z1 = torch.matmul(W1, x)
#
#             # Matrix multiplication to obtain the z score for each word
#             z2 = torch.matmul(W2, z1)
#
#             # Apply Log and softmax functions
#             log_softmax = F.log_softmax(z2, dim=0)
#             # Compute the negative-log-likelihood loss
#             loss = F.nll_loss(log_softmax.view(1, -1), y_true)
#             loss_val += loss.item()
#
#             # compute the gradient in function of the error
#             loss.backward()
#
#             # Update your embeddings
#             W1.data -= learning_rate * W1.grad.data
#             W2.data -= learning_rate * W2.grad.data
#
#             W1.grad.data.zero_()
#             W2.grad.data.zero_()
#
#         losses.append(loss_val / len(training_pairs))
#
#     return W1, W2, losses
#
# W1, W2, losses = Skip_Gram(training_pairs, word2idx, epochs=100)
#
#
# def plot_loss(loss):
#     x_axis = [epoch+1 for epoch in range(len(loss))]
#     plt.plot(x_axis, loss, '-g', linewidth=1, label='Train')
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.show()
#
# plot_loss(losses)


# # TFIDF
#
# tfidf_vectorizer = TfidfTransformer()
# tfidf_vectorizer.fit(X)
# # get feature names
# feature_names = cv.get_feature_names()
#
# # fetch document for which keywords needs to be extracted
# doc = cleaned_documents[5]  # 532
#
# # generate tf-idf for the given document
# tf_idf_vector = tfidf_vectorizer.transform(cv.transform([doc]))
#
# tf_idf_vector.toarray()
#
#
# def extract_feature_scores(feature_names, document_vector):
#     """
#     Function that creates a dictionary with the TF-IDF score for each feature.
#     :param feature_names: list with all the feature words.
#     :param document_vector: vector containing the extracted features for a specific document
#
#     :return: returns a sorted dictionary "feature":"score".
#     """
#     feature2score = {}
#     for i in range(len(feature_names)):
#         feature2score[feature_names[i]] = document_vector[0][i]
#     return sorted(feature2score.items(), key=lambda kv: kv[1], reverse=True)
#
# extract_feature_scores(feature_names, tf_idf_vector.toarray())[:10]


############ BAG OF WORDS #################

cv = CountVectorizer(max_df=0.9, binary=True)
X = cv.fit_transform(stem_file_data['text'])
y = np.array(stem_file_data["author"])

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



# # Instantiate the Portuguese model: nlp
# nlp = pt_core_news_sm.load()

# article = cleaned_documents[0]
# # Create a new document: doc
# doc = nlp(article)

# # Print all of the found entities and their labels
# for ent in doc.ents:
#     print(ent.label_, ent.text)


# Create a series to store the labels: y
y = file_data.author
file_data["author"] = file_data["author"].astype('category')

# Get the numerical ids of column label
numerical_ids = file_data.author.cat.codes
# Print initial shape
print(numerical_ids.shape)

# One-hot encode the indexes
Y = to_categorical(numerical_ids)
# Check the new shape of the variable
print(Y.shape)
# Print the first 5 rows
print(Y[:5])


# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 70000
# Max number of words in each text.
MAX_SEQUENCE_LENGTH = 5000
# This is fixed.
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
tokenizer.fit_on_texts(file_data.text.values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# Change text for numerical ids and pad
X = tokenizer.texts_to_sequences(file_data.text)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)
Y = pd.get_dummies(file_data['author']).values
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.1, random_state=42)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


epochs =2
batch_size = 24


history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,callbacks=[EarlyStopping(monitor='loss', patience=3, min_delta=0.0001)])


accr = model.evaluate(X_test,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    # model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(Dense(100, input_dim=MAX_SEQUENCE_LENGTH, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=25, verbose=0)
kfold = KFold(n_splits=5, shuffle=True)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

# Change text for numerical ids and pad
X_new = tokenizer.texts_to_sequences(file_data_test.text)
X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

# # One-hot encode the labels
# y_true= pd.get_dummies(file_data['author']).values
#

# Use the model to predict on new data
predicted = model.predict(X_new)

# Choose the class with higher probability
file_data_test["pred"] = np.argmax(predicted, axis=1)


# # Choose the class with higher probability
# y_pred = np.argmax(predicted, axis=1)

# # Compute and print the confusion matrix
# print(confusion_matrix(y_true, y_pred))
#
# # Create the performance report
# print(classification_report(y_true, y_pred, target_names=news_cat))



# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer()

# Transform the training data using only the 'text' column values: count_train
count_train = count_vectorizer.fit_transform(X_train)

# Transform the test data using only the 'text' column values: count_test
count_test = count_vectorizer.transform(X_test)

# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:10])


# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer( max_df=0.7)

# Transform the training data: tfidf_train
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data: tfidf_test
tfidf_test = tfidf_vectorizer.transform(X_test)

# Print the first 10 features
print(tfidf_vectorizer.get_feature_names()[:10])

# Print the first 5 vectors of the tfidf training data
print(tfidf_train.A[:5])

# Create the CountVectorizer DataFrame: count_df
count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

# Create the TfidfVectorizer DataFrame: tfidf_df
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

# Print the head of count_df
print(count_df.head())

# Print the head of tfidf_df
print(tfidf_df.head())

# Calculate the difference in columns: difference
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)

# Check whether the DataFrames are equal
print(count_df.equals(tfidf_df))


# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(count_train,y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test,pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test,pred)
print(cm)

# Create a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(tfidf_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(tfidf_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred)
print(cm)

# Create the list of alphas: alphas
alphas = np.arange(0,1,step=0.1)

# Define train_and_predict()
def train_and_predict(alpha):
    # Instantiate the classifier: nb_classifier
    nb_classifier = MultinomialNB(alpha=alpha)
    # Fit to the training data
    nb_classifier.fit(tfidf_train, y_train)
    # Predict the labels: pred
    pred = nb_classifier.predict(tfidf_test)
    # Compute accuracy: score
    score = metrics.accuracy_score(y_test,pred)
    return score

# Iterate over the alphas and print the corresponding score
for alpha in alphas:
    print('Alpha: ', alpha)
    print('Score: ', train_and_predict(alpha))


# Get the class labels: class_labels
class_labels = nb_classifier.classes_

# Extract the features: feature_names
feature_names = tfidf_vectorizer.get_feature_names()

# Zip the feature names together with the coefficient array and sort by weights: feat_with_weights
feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))

# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20])

# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[-20:])
