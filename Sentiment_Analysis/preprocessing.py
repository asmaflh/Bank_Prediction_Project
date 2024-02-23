from cleantext import clean
import re
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer

from nltk.corpus import stopwords
import jieba.posseg as pseg

stop_words_en = set(stopwords.words('english'))
stop_words_fr = set(stopwords.words('french'))
stop_words_it = set(stopwords.words('italian'))
stop_words_sp = set(stopwords.words('spanish'))
stop_words_de = set(stopwords.words('german'))
stop_words_cn = set(stopwords.words('chinese'))



def data_processing_en(text):
    text = text.lower()
    # remove urls
    text = re.sub(r"https\S+|www\S+https\S+", '', text, flags=re.MULTILINE)
    # remove numbers
    text = re.sub(r'\d+', '', text)
    # Removing User Mentions and Hashtags
    text = re.sub(r'@\w+|\#\w+', '', text)
    # Removing Punctuation:
    text = re.sub(r'[^\w\s]', '', text)
    # text = clean(text, no_emoji=True)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if len(w) >= 4 and not w in stop_words_en]
    stemmer = SnowballStemmer("english")
    stemmer_text = [stemmer.stem(word) for word in filtered_text]
    return " ".join(stemmer_text)


def data_processing_fr(text):
    text = text.lower()
    # remove urls
    text = re.sub(r"https\S+|www\S+https\S+", '', text, flags=re.MULTILINE)
    # remove numbers
    text = re.sub(r'\d+', '', text)
    # Removing User Mentions and Hashtags
    text = re.sub(r'@\w+|\#\w+', '', text)
    # Removing Punctuation:
    text = re.sub(r'[^\w\s]', '', text)
    # text = clean(text, no_emoji=True)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if len(w) >= 4 and not w in stop_words_fr]
    stemmer = SnowballStemmer("french")
    stemmer_text = [stemmer.stem(word) for word in filtered_text]
    return " ".join(stemmer_text)


def data_processing_it(text):
    text = text.lower()
    # remove urls
    text = re.sub(r"https\S+|www\S+https\S+", '', text, flags=re.MULTILINE)
    # remove numbers
    text = re.sub(r'\d+', '', text)
    # Removing User Mentions and Hashtags
    text = re.sub(r'@\w+|\#\w+', '', text)
    # Removing Punctuation:
    text = re.sub(r'[^\w\s]', '', text)
    # text = clean(text, no_emoji=True)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if len(w) >= 4 and not w in stop_words_it]
    stemmer = SnowballStemmer("italian")
    stemmer_text = [stemmer.stem(word) for word in filtered_text]
    return " ".join(stemmer_text)


def data_processing_es(text):
    text = text.lower()
    # remove urls
    text = re.sub(r"https\S+|www\S+https\S+", '', text, flags=re.MULTILINE)
    # remove numbers
    text = re.sub(r'\d+', '', text)
    # Removing User Mentions and Hashtags
    text = re.sub(r'@\w+|\#\w+', '', text)
    # Removing Punctuation:
    text = re.sub(r'[^\w\s]', '', text)
    # text = clean(text, no_emoji=True)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if len(w) >= 4 and not w in stop_words_sp]
    stemmer = SnowballStemmer("spanish")
    stemmer_text = [stemmer.stem(word) for word in filtered_text]
    return " ".join(stemmer_text)


def data_processing_de(text):
    text = text.lower()
    # remove urls
    text = re.sub(r"https\S+|www\S+https\S+", '', text, flags=re.MULTILINE)
    # remove numbers
    text = re.sub(r'\d+', '', text)
    # Removing User Mentions and Hashtags
    text = re.sub(r'@\w+|\#\w+', '', text)
    # Removing Punctuation:
    text = re.sub(r'[^\w\s]', '', text)
    # text = clean(text, no_emoji=True)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if len(w) >= 4 and not w in stop_words_de]
    stemmer = SnowballStemmer("german")
    stemmer_text = [stemmer.stem(word) for word in filtered_text]
    return " ".join(stemmer_text)


def data_processing_cn(text):
    # remove urls
    text = re.sub(r"https\S+|www\S+https\S+", '', text, flags=re.MULTILINE)
    # remove numbers
    text = re.sub(r'\d+', '', text)
    # Removing User Mentions and Hashtags
    text = re.sub(r'@\w+|\#\w+', '', text)
    # Removing Punctuation:
    text = re.sub(r'[^\w\s]', '', text)
    # text = clean(text, no_emoji=True)
    text_tokens =list(text)
    filtered_text = [w for w in text_tokens if not w in stop_words_cn]
    return " ".join(filtered_text)
