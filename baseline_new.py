import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
import os
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

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
    """
    Function that a receives a list of strings and preprocesses it.

    :param text_list: List of strings.
    :param lemmatize: Tag to apply lemmatization if True.
    :param stemmer: Tag to apply the stemmer if True.
    """
    processed_corpus = []
    for i in range(len(dataframe)):
        text = dataframe['text'][i]
        # LOWERCASE TEXT
        text = text.lower()

        # REMOVE NUMERICAL DATA AND PUNCTUATION
        text = re.sub("[^a-zA-Z-ÁÀÂÃâáàãçÉÈÊéèêúùÚÙÕÓÒÔôõóòÍÌíìçÇ]", ' ', text)

        # REMOVE TAGS
        text = BeautifulSoup(text).get_text()
        text = text.split()
        # REMOVE STOP WORDS
        text = [lemma.lemmatize(word) for word in text if not word in stop]

        text = " ".join(text)
        processed_corpus.append(text)

    return processed_corpus


cleaned_documents = preprocessing(file_data)
cleaned_documents_test = preprocessing(file_data_test)


def update_df(dataframe, cleaned_documents):
    dataframe['text'] = cleaned_documents


update_df(file_data, cleaned_documents)
update_df(file_data_test, cleaned_documents_test)

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

cv = CountVectorizer(
    max_df=0.8,
    #max_features=10000,
    #ngram_range=(1,3) # only bigram (2,2)
)

X = cv.fit_transform(cleaned_documents)

vocab = list(cv.vocabulary_.keys())

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
top_df_test = get_top_n_grams(cleaned_documents_test,top_k = 20, n = 2)
top_df.head(10)
top_df_test.head(10)
plot_frequencies(top_df)
plot_frequencies(top_df_test)

# TFIDF

tfidf_vectorizer = TfidfTransformer()
X_train_tf_idf = tfidf_vectorizer.fit_transform(X)
# generate tf-idf for the given document
X_pred = tfidf_vectorizer.transform(cv.transform(cleaned_documents_test))

# get feature names
feature_names = cv.get_feature_names()
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tf_idf, file_data.author)
predicted = clf.predict(X_pred)

#score = np.mean(predicted == file_data_test.target)
# fetch document for which keywords needs to be extracted
#doc = cleaned_documents[5]  # 532

