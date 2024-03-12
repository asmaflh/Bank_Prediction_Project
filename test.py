import os
import re
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import pandas as pd
import argparse
import nltk
from textblob import Blobber
import unicodedata
from textblob_fr import PatternTagger, PatternAnalyzer
from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, Layer, Activation, Dropout, concatenate, \
    Flatten, MaxPooling1D, AveragePooling1D, Conv1D
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras import backend as K
import numpy as np
import pickle
import spacy
import re
import os

from Sentiment_Analysis.preprocessing import data_processing_it

path = os.path.dirname(__file__)

MAX_SEQUENCE_LENGTH = 35
EMBEDDING_DIM = 300
MAX_N_WEMBS = 200000
nlp = spacy.load("it_core_news_sm")
NB_WEMBS = MAX_N_WEMBS

with open(os.path.join(path, 'wemb_ind.pkl'), 'rb') as f:
    wemb_ind = pickle.load(f)


def tok_process(doc):
    doc_toks = list(map(lambda x: x.text.lower(), doc))
    return doc_toks


def create_word_list(texts):
    docs = list(map(lambda x: nlp(x), texts))
    wlist = list(map(lambda x: tok_process(x), docs))
    word_list = [w for sublist in wlist for w in sublist]
    return list(set(word_list))


def process_texts(texts, maxlen):
    texts_feats = map(lambda x: create_features(x, maxlen), texts)
    return texts_feats


def create_features(text, maxlen):
    doc = nlp(text)

    wemb_idxs = []
    for tok in doc:
        text_lower = tok.text.lower()
        wemb_idx = wemb_ind.get(text_lower, wemb_ind['unknown'])
        wemb_idxs.append(wemb_idx)
    wemb_idxs = pad_sequences([wemb_idxs], maxlen=maxlen, value=NB_WEMBS)
    return [wemb_idxs]


# ================ Model ======================#
def load_model(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):
    max_len = MAX_SEQUENCE_LENGTH
    emb_dim = EMBEDDING_DIM

    x1 = Input(shape=(max_len,))
    w_embed = Embedding(NB_WEMBS + 1, emb_dim, input_length=max_len, trainable=False)(x1)
    w_embed = Dropout(0.5)(Dense(64, activation='relu')(w_embed))
    h = Conv1D(filters=32, kernel_size=2, padding='valid', activation='relu')(w_embed)
    h = Bidirectional(LSTM(32, return_sequences=True, recurrent_dropout=0.5), merge_mode='concat')(h)
    h = AveragePooling1D(pool_size=2, strides=None, padding='valid')(h)
    h = Bidirectional(LSTM(16, return_sequences=True, recurrent_dropout=0.5), merge_mode='concat')(h)
    h = AveragePooling1D(pool_size=2, strides=None, padding='valid')(h)
    h = Flatten()(h)
    preds_pol = Dense(2, activation='sigmoid')(h)

    model_pol = Model(inputs=[x1], outputs=preds_pol)
    model_pol.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    # model_pol.summary()

    STAMP = 'test_sentita_lstm-cnn_wikiner_v1'
    # early_stopping = EarlyStopping(monitor='val_loss', patience=7)
    # bst_model_path = os.path.join(path, STAMP + '.h5')
    # checkpointer = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
    model_pol.load_weights("test_sentita_lstm-cnn_wikiner_v1.h5")
    return model_pol


def calculate_polarity(sentences):
    results = []
    sentences = list(map(lambda x: x.lower(), sentences))
    # sentences = list(map(lambda x: re.sub('[^a-zA-z0-9\s]','',x), sentences))
    X_ctest = list(process_texts(sentences, MAX_SEQUENCE_LENGTH))
    n_ctest_sents = len(X_ctest)
    #
    test_wemb_idxs = np.reshape(np.array([e[0] for e in X_ctest]), [n_ctest_sents, MAX_SEQUENCE_LENGTH])
    #
    sent_model = load_model(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
    preds = sent_model.predict([test_wemb_idxs])
    K.clear_session()
    for i in range(n_ctest_sents):
        try:
            if preds[i][0] >= preds[i][1]:
                pol = preds[i][0]
            else:
                pol = preds[i][1]
        except Exception:

            pol = 0

        results.append(0 if np.isnan(pol) else pol)
    return results


df = pd.read_csv("Twitter_Datasets/TwitterData_IT.csv", encoding='utf8')
df.Tweet = df['Tweet'].apply(data_processing_it)
df = df.drop_duplicates('Tweet')

ListOfTweets = df['Tweet']

results = calculate_polarity(ListOfTweets)

df["Polarity"] = pd.Series(results)


def check_if_nan(a):
    return 0 if np.isnan(a) else a


df["Polarity"] = df["Polarity"].apply(check_if_nan)
print(df['Polarity'])