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


def preprocessing(dataframe):
    
    processed_corpus = []
    for i in range(len(dataframe)):
        text = dataframe['text'][i]
        #LOWERCASE TEXT
        text = text.lower()
        
        #REMOVE NUMERICAL DATA AND PUNCTUATION
        text = re.sub("[^a-zA-Z-ÁÀÂÃâáàãçÉÈÊéèêúùÚÙÕÓÒÔôõóòÍÌíìçÇ]", ' ', text)
        
        #REMOVE TAGS
        text = BeautifulSoup(text).get_text()
        text = text.split()
        
        #REMOVE STOP WORDS
        text = [lemma.lemmatize(word) for word in text if not word in stop]

        text = " ".join(text)
        processed_corpus.append(text)
        
    return processed_corpus

cleaned_documents= preprocessing(file_data)
cleaned_documents_test=preprocessing(file_data_test)

def update_df(dataframe, cleaned_documents):
    dataframe['text'] = cleaned_documents


update_df(file_data, cleaned_documents)
update_df(file_data_test,cleaned_documents_test)

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

top_df2 = get_top_n_grams(cleaned_documents, top_k=10, n=2)
top_df3 = get_top_n_grams(cleaned_documents, top_k=10, n=3)

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

X = cv.fit_transform(cleaned_documents)
X_new = cv.fit_transform(cleaned_documents_test)

list(cv.vocabulary_.keys())[:10]

tfidf_vectorizer = TfidfTransformer()
tfidf_vectorizer.fit(X)
# get feature names
feature_names = cv.get_feature_names()

# fetch document for which keywords needs to be extracted
doc = cleaned_documents[10]  # 532

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
X = cv.fit_transform(file_data['text'])
y = np.array(file_data["author"])

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
        """
        Initializes a matrix in which each column will be the Weights for a specific class.
        :param input_size: Number of features
        :param n_classes: Number of classes to classify the inputs
        """
        self.parameters = np.zeros((input_size + 1, n_classes))  # input_size +1 to include the Bias term

    def train(self, X, Y, devX, devY, epochs=20):
        """
        This trains the perceptron over a certain number of epoch and records the
            accuracy in Train and Dev sets along each epoch.
        :param X: numpy array with size DxN where D is the number of training examples
                 and N is the number of features.
        :param Y: numpy array with size D containing the correct labels for the training set
        :param devX (optional): same as X but for the dev set.
        :param devY (optional): same as Y but for the dev set.
        :param epochs (optional): number of epochs to run.
        """
        train_accuracy = [self.evaluate(X, Y)]
        dev_accuracy = [self.evaluate(devX, devY)]
        for epoch in range(epochs):
            for i in tqdm(range(X.shape[0])):
                self.update_weights(X[i, :], Y[i])
            train_accuracy.append(self.evaluate(X, Y))
            dev_accuracy.append(self.evaluate(devX, devY))
        return train_accuracy, dev_accuracy

    def evaluate(self, X, Y):
        """
        Evaluates the error in a given set of examples.
        :param X: numpy array with size DxN where D is the number of examples to
                    evaluate and N is the number of features.
        :param Y: numpy array with size D containing the correct labels for the training set
        """
        correct_predictions = 0
        for i in range(X.shape[0]):
            y_pred = self.predict(X[i, :])
            if Y[i] == y_pred:
                correct_predictions += 1
        return correct_predictions / X.shape[0]

    def plot_train(self, train_accuracy, dev_accuracy):
        """
        Function to Plot the accuracy of the Training set and Dev set per epoch.
        :param train_accuracy: list containing the accuracies of the train set.
        :param dev_accuracy: list containing the accuracies of the dev set.
        """
        x_axis = [epoch + 1 for epoch in range(len(train_accuracy))]
        plt.plot(x_axis, train_accuracy, '-g', linewidth=1, label='Train')
        plt.xlabel("epochs")
        plt.ylabel("Accuracy")
        plt.plot(x_axis, dev_accuracy, 'b-', linewidth=1, label='Dev')
        plt.legend()
        plt.show()

    def update_weights(self, x, y):
        """
        Function that will take an input example and the true prediction and will
            update the model parameters.
        :param x: Array of size N where N its the number of features that the model
                  takes as input.
        :param y: The int corresponding to the correct label.
        child classes must implement this function
        """
        pass

    def predict(self, x):
        """
        This function will add a Bias value to the received input, multiply the Weights
            corresponding to the different classeswith the input vector and choose the
            class that maximizes that multiplication.
        :param x: numpy array with size 1xN where N = number of features.
        child classes must implement this function
        """
        pass



class MultinomialLR(Classifier):
    """ Multinomial Logistic Regression """

    def __init__(self, input_size, n_classes, lr=0.001):
        """
        Initializes a matrix in which each column will be the Weights for a specific class.
        :param input_size: Number of features
        :param n_classes: Number of classes to classify the inputs
        """
        Classifier.__init__(self, input_size, n_classes)
        self.lr = lr

    def predict(self, input):
        """
        This function will add a Bias value to the received input, multiply the
            Weights corresponding to the different classeswith the input vector, run
            a softmax function and choose the class that achieves an higher probability.
        :param x: numpy array with size 1xN where N = number of features.
        """
        return np.argmax(self.softmax(np.dot(np.append(input, [1]), self.parameters)))

    def softmax(self, x):
        """ Compute softmax values for each sets of scores in x."""
        return np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)), axis=0)

    def update_weights(self, x, y):
        """
        Function that will take an input example and the true prediction and will update
            the model parameters.
        :param x: Array of size N where N its the number of features that the model takes as input.
        :param y: The int corresponding to the correct label.
        """
        linear = np.dot(np.append(x, [1]), self.parameters)
        predictions = self.softmax(linear)
        self.parameters = self.parameters - self.lr*(np.outer(predictions, np.append(x, [1])).T)
        self.parameters[:, y] = self.parameters[:, y] + self.lr*np.append(x, [1])


lr  = MultinomialLR(X.shape[1], len(np.unique(Y)))

print(lr.parameters.shape)

X_test = vectorizer.transform(test.text).toarray()
Y_test = test.author



train_acc, dev_acc = lr.train(X, Y, devX=X_test, devY=Y_test, epochs=5)

lr.plot_train(train_acc, dev_acc)

#
#
# #Random code
# # Create a Dictionary from the articles: dictionary
# dictionary = Dictionary(file_data['text'].to_array())
#
# # Select the id for "computer": computer_id
# computer_id = file_name_and_text.token2id.get("computer")
#
# # Use computer_id with the dictionary to print the word
# print(dictionary.get(computer_id))
#
# # Create a MmCorpus: corpus
# corpus = [dictionary.doc2bow(article) for article in articles]
#
# # Print the first 10 word ids with their frequency counts from the fifth document
# print(corpus[4][:10])
#
# # Save the fifth document: doc
# doc = corpus[4]
#
# # Sort the doc for frequency: bow_doc
# bow_doc = sorted(doc, key=lambda w: w[1], reverse=True)
#
# # Print the top 5 words of the document alongside the count
# for word_id, word_count in bow_doc[:5]:
#     print(dictionary.get(word_id), word_count)
#
# # Create the defaultdict: total_word_count
# total_word_count = defaultdict(int)
# for word_id, word_count in itertools.chain.from_iterable(corpus):
#     total_word_count[word_id] += word_count
#
# # Create a sorted list from the defaultdict: sorted_word_count
# sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True)
#
# # Print the top 5 words across all documents alongside the count
# for word_id, word_count in sorted_word_count[:5]:
#     print(dictionary.get(word_id), word_count)
#
#
# # Create a new TfidfModel using the corpus: tfidf
# tfidf = TfidfModel(corpus)
#
# # Calculate the tfidf weights of doc: tfidf_weights
# tfidf_weights = tfidf[doc]
#
# # Print the first five weights
# print(tfidf_weights[:5])
#
# # Sort the weights from highest to lowest: sorted_tfidf_weights
# sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)
#
# # Print the top 5 weighted words
# for term_id, weight in sorted_tfidf_weights[:5]:
#     print(dictionary.get(term_id), weight)
#
# # Tokenize the article into sentences: sentences
# sentences = nltk.sent_tokenize(article)
#
# # Tokenize each sentence into words: token_sentences
# token_sentences = [nltk.word_tokenize(sent) for sent in sentences]
#
# # Tag each tokenized sentence into parts of speech: pos_sentences
# pos_sentences = [ nltk.pos_tag(sent) for sent in token_sentences]
#
# # Create the named entity chunks: chunked_sentences
# chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)
#
# # Test for stems of the tree with 'NE' tags
# for sent in chunked_sentences:
#     for chunk in sent:
#         if hasattr(chunk, "label") and chunk.label() == "NE":
#             print(chunk)
#
# # Create the defaultdict: ner_categories
# ner_categories = defaultdict(int)
# # Create the nested for loop
# for sent in chunked_sentences:
#     for chunk in sent:
#         if hasattr(chunk, 'label'):
#             ner_categories[chunk.label()] += 1
#
# # Create a list from the dictionary keys for the chart labels: labels
# labels = list(ner_categories.keys())
#
# # Create a list of the values: values
# values = [ner_categories.get(l) for l in labels]
#
# # Create the pie chart
# plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
#
# # Display the chart
# plt.show()


# Instantiate the Portuguese model: nlp
nlp = pt_core_news_sm.load()

article = cleaned_documents[0]
# Create a new document: doc
doc = nlp(article)

# Print all of the found entities and their labels
for ent in doc.ents:
    print(ent.label_, ent.text)


# Create a series to store the labels: y
y = file_data.author
file_data["author"] = file_data["author"].astype('category')

#from keras.utils.np_utils import to_categorical
#from keras.models import Sequential
#from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
#from keras.callbacks import EarlyStopping
#from keras.layers import Dropout

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 100000
# Max number of words in each text.
MAX_SEQUENCE_LENGTH = 10000
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
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)

print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs =5
batch_size = 2

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


accr = model.evaluate(X_test,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


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
#
#


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