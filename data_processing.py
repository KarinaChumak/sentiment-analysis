import pandas as pd
import nltk
import functools

porter = nltk.PorterStemmer()
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
import re
from custom_stopwords import stopwords, negative_words


def filter_negatives(word):
    if word in stopwords:
        return ''
    elif word in negative_words:
        return 'not'
    else:
        return word

def text_preprocess(text, drop_numbers, replace_negatives, stem):
    text = re.sub(r'\t0|\t1|\s', ' ', text)
    text = re.sub('[?.,\/#!$%\^&\*;:{}=\-_`~()]','', text.lower())
    if drop_numbers:
        text = re.sub('\d+', '', text)
    if replace_negatives:
        words = [filter_negatives(word) for word in text.split()]

        # words = [word for word in text.split()]
    else:
        words = [word for word in text.split() if word not in stop_words]

    if stem:
        words = list(map(porter.stem, words))

    return re.sub(r'\s+', ' ', ' '.join(words))


def load_process_data(dataset_name = 'amazon',drop_numbers = True, replace_negatives = True, stem = False):
    if dataset_name == 'amazon':
        filename = 'data/amazon_cells_labelled.txt'
    elif dataset_name == 'yelp':
        filename = 'data/yelp_labelled.txt'
    elif dataset_name == 'imdb':
        filename = 'data/imdb_labelled.txt'
    else:
        print("Wrong dataset name")
    data = pd.read_csv(filename, delimiter = '\t', names = ['text','label'])

    process = functools.partial(text_preprocess, drop_numbers= drop_numbers, replace_negatives = replace_negatives, stem=stem)
    data['processed_text'] = data['text'].apply(process)
    #outliers
    data = data[data['processed_text'].apply(lambda x : len(x.split())) <= 35]
    return data

        