# Import the necessary modules
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from bs4 import BeautifulSoup
import string
import pandas as pd
import numpy as np
import os
import re
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
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
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import warnings
import datetime
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=DeprecationWarning)


#import the train scripts
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

file_data_raw = (pd.DataFrame.from_dict(file_name_and_text, orient='index')
             .reset_index().rename(index = str, columns = {'index': 'author', 0: 'text'}))

#import the test scripts
root_test = os.getcwd() + '\\Corpora\\test'

file_name_and_text_new={}
for file in os.listdir(root_test):
    name = os.path.splitext(file)[0]
    print('{}/{}'.format(root_test,name))

    for file in os.listdir('{}/{}'.format(root_test,name)):
        if file.endswith(".txt"):
            try:
                with open(os.path.join(root_test, name,file), "r", encoding='utf-8') as target_file:
                    file_name_and_text_new['{}/{}'.format(name,file)] = target_file.read()
            except Exception as e:
                print("{} generated an error: \n {}".format(os.path.join(root_test, name,file)),e)

file_data_new = (pd.DataFrame.from_dict(file_name_and_text_new, orient='index')
             .reset_index().rename(index = str, columns = {'index': 'number_of_words', 0: 'text'}))

#split the titles into n word samples

def split_strings_n_words(df, n):
    new_df = pd.concat([pd.Series(row['author'], [' '.join(row['text'].split()[x:x + n]) for x in range(0, len(row['text'].split()), n)]) for _, row in df.iterrows()]).reset_index()
    # new data frame with split value columns
    new = new_df[0].str.split("/", n=1, expand=True)
    new_df.drop(columns=[0], inplace=True)
    new_df["author"] = new[0]
    new_df["title"] = new[1]
    new_df.rename(columns={"index":"text"}, inplace=True)
    return new_df

file_data = split_strings_n_words(file_data_raw,500)
file_data_1000 = split_strings_n_words(file_data_raw,1000)

#uncomment to check the imbalanced sampling
# file_data.groupby(['author','title']).count().to_csv("number_of_500_samples_per_title_per_author.csv")
# file_data_1000.groupby(['author','title']).count().to_csv("number_of_1000_samples_per_title_per_author.csv")

#initialise preprocessing parameters
stop = set(stopwords.words('portuguese'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
stemmer = SnowballStemmer('portuguese')

#basic preprocessing function for NN
def clean_data(dataframe):

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


def update_df(dataframe, cleaned_documents):
    dataframe['text'] = cleaned_documents

#create test and train samples for both 500 word samples split and 1000 word sample split
#these titles were chosen because their wordcount corresponded to aprox. 10% of the words for each author

test=['pg22801.txt','pg27364.txt','pg25641.txt','O Setimo Selo - Jose Rodrigues dos Santos.txt','O Homem Duplicado - Jose Saramago.txt','ABelaHistoria.txt']


cleaned_documents = clean_data(file_data)
update_df(file_data, cleaned_documents)
file_data_subset = file_data.loc[~file_data.title.isin(test)].reset_index()
file_data_test = file_data.loc[file_data.title.isin(test)].reset_index()
y_train = np.array(file_data['author'].loc[~file_data.title.isin(test)])
y_test = np.array(file_data['author'].loc[file_data.title.isin(test)])

#1000 split
cleaned_documents_1000= clean_data(file_data_1000)
update_df(file_data_1000, cleaned_documents_1000)

file_data_subset_1000 = file_data_1000.loc[~file_data_1000.title.isin(test)].reset_index()
file_data_test_1000 = file_data_1000.loc[file_data_1000.title.isin(test)].reset_index()
y_train_1000 = np.array(file_data_1000['author'].loc[~file_data_1000.title.isin(test)])
y_test_1000 = np.array(file_data_1000['author'].loc[file_data_1000.title.isin(test)])

#new data to predict on including tentative labels obtained from googling the text excerpts to be predicted
cleaned_documents_new=clean_data(file_data_new)
update_df(file_data_new,cleaned_documents_new)
file_data_new.insert(1,'y_true',['JoseSaramago','AlmadaNegreiros','LuisaMarquesSilva','EcaDeQueiros','CamiloCasteloBranco','JoseRodriguesSantos','JoseSaramago','AlmadaNegreiros','LuisaMarquesSilva','EcaDeQueiros','CamiloCasteloBranco','JoseRodriguesSantos'])
file_data_new.insert(2,'predicted',value=None)
file_data_new.insert(3,'pred_KNN',value=None)
file_data_new.insert(4,'pred_NB',value=None)
file_data_new.insert(5,'predicted_500',value=None)
file_data_new.insert(6,'predicted_1000',value=None)


#undersampling to balance the dataset
def balancing_undersampling(df):
    print("Before balanced resampling:")
    print(df.groupby(['author'])['text'].count())
    g = df.groupby('author')
    balanced_df = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()))).reset_index(drop=True)
    print("After balanced resampling:")
    print(balanced_df.groupby(['author'])['text'].count())
    return balanced_df

#using holdout test and train
balanced_train=balancing_undersampling(file_data_subset)
balanced_test = balancing_undersampling(file_data_test)

balanced_train_1000=balancing_undersampling(file_data_subset_1000)
balanced_test_1000 = balancing_undersampling(file_data_test_1000)

#full dataset balancing
balanced=balancing_undersampling(file_data)
balanced_1000= balancing_undersampling(file_data_1000)

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

def generate_stem_df(train_df,test_df,new_df):

    stem_documents = stem_stop_words(train_df)
    stem_documents_test = stem_stop_words(test_df)
    stem_documents_new = stem_stop_words(new_df)

    stem_file_data = train_df.copy(deep=True)
    stem_file_data_test = test_df.copy(deep=True)
    stem_file_data_new = new_df.copy(deep=True)
    stem_file_data['text'] = stem_documents
    stem_file_data_test['text'] = stem_documents_test
    stem_file_data_new['text'] = stem_documents_new

    return stem_file_data,stem_file_data_test,stem_file_data_new

#uses 500 word sample split by default

stem_file_data,stem_file_data_test,stem_file_data_new = generate_stem_df(file_data_subset,file_data_test,file_data_new)

###lemma_file_data and lemma_documents also has lemmatisation and stopwords removed##
#to be used for NaiveBayes etc retains less text info##

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

def generate_lemma_df(train_df,test_df,new_df):

    lemma_documents = lemma_stop_words(train_df)
    lemma_documents_test = lemma_stop_words(test_df)
    lemma_documents_new = lemma_stop_words(new_df)

    lemma_file_data = train_df.copy(deep=True)
    lemma_file_data_test = test_df.copy(deep=True)
    lemma_file_data_new = new_df.copy(deep=True)
    lemma_file_data['text'] = lemma_documents
    lemma_file_data_test['text'] = lemma_documents_test
    lemma_file_data_new['text'] = lemma_documents_new

    return lemma_file_data,lemma_file_data_test,lemma_file_data_new

#uses 500 word sample split by default
#using holdout train test
lemma_file_data,lemma_file_data_test,lemma_file_data_new = generate_lemma_df(file_data_subset,file_data_test,file_data_new)
#using full dataset
lemma_file_data_KNN,lemma_file_data_test,lemma_file_data_new = generate_lemma_df(file_data,file_data_test,file_data_new)
#using full balanced dataset
lemma_file_data_balanced,lemma_file_data_test,lemma_file_data_new =generate_lemma_df(balanced,file_data_test,file_data_new)

################## KNN ######################

def KNN(train_df, test_df):
    cv_KNN = CountVectorizer(
        max_df=0.8,
        max_features=1000,
        ngram_range=(1, 3)
    )

    tfidf_vectorizer = TfidfTransformer()

    X_KNN = tfidf_vectorizer.fit_transform(cv_KNN.fit_transform(train_df['text']))

    y = np.array(train_df["author"])

    X_train, X_test, y_train, y_test = train_test_split(X_KNN, y, test_size=0.2, random_state=42)

    modelknn = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='brute', leaf_size=30, p=2,
                                    metric='cosine', metric_params=None, n_jobs=1)
    modelknn.fit(X_train, y_train)

    cv_KNN_test = CountVectorizer(vocabulary=cv_KNN.vocabulary_)
    X_KNN_new = tfidf_vectorizer.fit_transform(cv_KNN_test.fit_transform(test_df['text']))

    predKNN = modelknn.predict(X_test)
    print(classification_report(y_test, predKNN))

    predKNN_new = modelknn.predict(X_KNN_new)
    file_data_new["pred_KNN"] = predKNN_new
    print(classification_report(test_df['y_true'], predKNN_new))

    return predKNN_new

#using full balanced dataset and 20/80 test/train split
KNN(lemma_file_data_balanced, lemma_file_data_new)  # could change for stem dfs but worst results


# ################ Naive Bayes #################

def NB(train_df, test_df):
    cv_NB = CountVectorizer(
        max_df=0.8,
        max_features=1000,
        ngram_range=(1, 3)
    )

    tfidf_vectorizer = TfidfTransformer()

    X_NB = tfidf_vectorizer.fit_transform(cv_NB.fit_transform(train_df['text']))

    y = np.array(train_df["author"])
    X_train, X_test, y_train, y_test = train_test_split(X_NB, y, test_size=0.2, random_state=42, shuffle=True,
                                                        stratify=y) #30

    nb_classifier = MultinomialNB()

    nb_classifier.fit(X_train, y_train)

    cv_NB_test = CountVectorizer(vocabulary=cv_NB.vocabulary_)
    X_NB_new = tfidf_vectorizer.fit_transform(cv_NB_test.fit_transform(test_df['text']))

    predNB = nb_classifier.predict(X_test)
    print(classification_report(y_test, predNB))

    predNB_new = nb_classifier.predict(X_NB_new)
    file_data_new["pred_NB"] = predNB_new
    print(classification_report(test_df["y_true"], predNB_new))

    return predNB_new

#using full balanced dataset and 20/80 test/train split
NB(lemma_file_data_balanced, lemma_file_data_new)

#EMBEDDING size tune

def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens

tokenized_corpus = tokenize_corpus(balanced_test.text)
vocabulary = {word for doc in tokenized_corpus for word in doc}
print("corpora vocab length:{}".format(len(vocabulary)))


oov_tok = '<OOV>'
trunc_type = 'post'
padding_type = 'post'
embedding_dim=100
#set random seeds to obtain more or less reproducible results
random_seed=42
tf.random.set_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

#LSTM model definition

def LSTM_model(train_df,test_df,new_df,MAX_LEN,MAX_NB_WORDS,epochs,batch_size):

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True) # The maximum number of words to be used. (most frequent) or could use whole vocab size
    tokenizer.fit_on_texts(train_df.text.values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # Change text for numerical ids and pad
    X = tokenizer.texts_to_sequences(train_df.text)
    X = pad_sequences(X, maxlen=MAX_LEN) #max length of texts, used for padding
    print('Shape of data tensor:', X.shape)
    Y = pd.get_dummies(train_df['author'])
    y_test =pd.get_dummies(test_df['author'].loc[test_df.title.isin(test)])

    X_train, X_dev, y_train, y_dev = train_test_split(X, Y, test_size=0.01, random_state=random_seed, stratify=Y) #test_size held small as holdout used instead

    # Change text for numerical ids and pad
    X_test = tokenizer.texts_to_sequences(test_df.text)
    X_test = pad_sequences(X_test, maxlen=MAX_LEN)
    print(X_train.shape, y_train.shape)
    print(X_dev.shape, y_dev.shape)

    # kfold_splits=10
    # # # Instantiate the cross validator
    # skf = StratifiedKFold(n_splits=kfold_splits, shuffle=True,random_state=7)
    #
    # # # Loop through the indices the split() method returns
    #
    # for index, (train_indices, val_indices) in enumerate(skf.split(X, Y.values.argmax(1))):
    #     print
    #     "Training on fold " + str(index + 1) + "/10..."
    #     # Generate batches from indices
    #     X_train, X_dev = X[train_indices], X[val_indices]
    #     y_train, y_dev = Y[train_indices], Y[val_indices]

    # Clear model, and create it
    #v1
    model = Sequential()
    model.add(Embedding(input_dim=MAX_NB_WORDS, output_dim=100, input_length=X.shape[1]))

    #v2
    model.add(SpatialDropout1D(0.2))
    #model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(50))
    # #v3
    # # Add an Embedding layer expecting input , and output embedding dimension of size 100 we set at the top
    # model.add(tf.keras.layers.Embedding(MAX_NB_WORDS,embedding_dim,input_length=X.shape[1]))
    # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)))
    # # use ReLU in place of tanh function since they are very good alternatives of each other.
    # model.add(tf.keras.layers.Dense(embedding_dim, activation='relu'))

    #output layer
    # Add a Dense layer with 6 units and softmax activation.When we have multiple outputs, softmax convert outputs layers into a probability distribution.

    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_data=(X_test,y_test),callbacks=[EarlyStopping(monitor='loss', patience=3, min_delta=0.05)])

    # accuracy_history = history.history['acc']
    # val_accuracy_history = history.history['val_acc']
    # print( "Last training accuracy: " + str(accuracy_history[-1]) + ", last validation accuracy: " + str(val_accuracy_history[-1]))

    # save the model to disk
    # filename = 'lstm_model_{}_{}.pkl'.format(MAX_LEN,datetime.datetime.today().strftime("%d_%m_%Y_%H_%M_%S"))
    # pickle.dump(model, open(filename, 'wb'))
    # # # load the model from disk
    # model_name='lstm_model_1000_06_04_2020_11_50_20.pkl'
    # loaded_model = pickle.load(open(model_name, 'rb'))
    # evaluate the model
    train_acc = model.evaluate(X_train, y_train, verbose=0)

    test_acc = model.evaluate(X_test, y_test, verbose=0)
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

    # # Choose the class with higher probability
    new_df['predicted']=Y.columns[list(np.argmax(predicted, axis=1))]

    # Create the performance report
    print(classification_report(new_df['y_true'],Y.columns[list(np.argmax(predicted, axis=1))], target_names=Y.columns))
    return predicted

#with 1000 sample dataset
# pred=LSTM_model(balanced_train_1000, balanced_test_1000, file_data_new,1000,30000,10,32)

#with 500 sample dataset, parameters for the results presented in the report
pred=LSTM_model(file_data_subset, file_data_test, file_data_new,1000,100000,10,40)

