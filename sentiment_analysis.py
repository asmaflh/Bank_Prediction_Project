import gensim as gensim
import pandas as pd
import nltk
# nltk.download('vader_lexicon')
import numpy as np
import re
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style
from nltk import WordNetLemmatizer, PorterStemmer

style.use('ggplot')
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from xgboost import XGBClassifier

# read csv file
df = pd.read_csv('TwitterData.csv')

# display infos about the dataset
print(df.head())
print(df.info())
print(df.isnull().sum)
print(df.columns)
print(df['Location'].unique())
for col in df:
    print(df[col].unique())

# extract tweet column
text_df = df.drop(['Unnamed: 0', 'TweetId', 'UserId', 'UserUsername', 'Date', 'Location', 'Media',
                   'Likes', 'Retweets', 'Comments'], axis=1)

print(text_df['Tweet'].iloc[0], "\n")
print(text_df['Tweet'].iloc[1], "\n")
print(text_df['Tweet'].iloc[2], "\n")
print(text_df['Tweet'].iloc[3], "\n")
print(text_df['Tweet'].iloc[4], "\n")

print(text_df.info())
text_df=text_df.sample(n=3000)

# clean the dataset
def data_processing(text):
    text = text.lower()
    # remove urls
    text = re.sub(r"https\S+|www\S+https\S+", '', text, flags=re.MULTILINE)
    # remove numbers
    text = re.sub(r'\d+', '', text)
    # Removing User Mentions and Hashtags
    # text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'@\w+|\#\w+', '', text)
    # Removing Punctuation:
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = word_tokenize(text)
    # filtered_text = [w for w in text_tokens if not w in stop_words]
    filtered_text = [w for w in text_tokens if len(w) >= 4 and not w in stop_words]
    stemmer = PorterStemmer()
    stemmer_text = [stemmer.stem(word) for word in filtered_text]
    # lemmatizer = WordNetLemmatizer()
    # lemmatized_text = [lemmatizer.lemmatize(token) for token in filtered_text]
    return " ".join(stemmer_text)


text_df.Tweet = text_df['Tweet'].apply(data_processing)
print(text_df.head())
print(text_df['Tweet'].iloc[0], "\n")
print(text_df['Tweet'].iloc[1], "\n")
print(text_df['Tweet'].iloc[2], "\n")
print(text_df['Tweet'].iloc[3], "\n")
print(text_df['Tweet'].iloc[4], "\n")
# drop duplicate tweets
text_df = text_df.drop_duplicates('Tweet')

# def polarity(text):
#     return TextBlob(text).sentiment.polarity
# Create a SentimentIntensityAnalyzer object
analyzer = SentimentIntensityAnalyzer()


# Define a function to calculate polarity using VADER
def polarity(text):
    # Use the polarity_scores() method of the analyzer object to get sentiment scores
    sentiment_scores = analyzer.polarity_scores(text)
    # Return the polarity score
    return sentiment_scores['compound']  # 'compound' score represents overall sentiment polarity


text_df['Polarity'] = text_df['Tweet'].apply(polarity)

print(text_df.head(10))


def sentiment(label):
    if label <= -0.05:
        return "Negative"
    elif label >= 0.05:
        return "Positive"
    else:
        return "Neutral"


text_df['Sentiment'] = text_df['Polarity'].apply(sentiment)

print(text_df.head())
# plot the sentiments
fig = plt.figure(figsize=(5, 5))
sns.countplot(x='Sentiment', data=text_df)

fig = plt.figure(figsize=(7, 7))
colors = ("yellowgreen", "gold", "red")
wp = {'linewidth': 2, 'edgecolor': "black"}
tags = text_df['Sentiment'].value_counts()
explode = (0.1, 0.1, 0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors,
          startangle=90, wedgeprops=wp, explode=explode, label='')

plt.title('Distribution of sentiments')

# Sorting Positive Tweets by Polarity
pos_tweets = text_df[text_df.Sentiment == 'Positive']
pos_tweets = pos_tweets.sort_values(['Polarity'], ascending=False)
print(pos_tweets['Tweet'].head(10))

# show most frequent words in positive tweets
text = ' '.join([word for word in pos_tweets['Tweet']])
plt.figure(figsize=(20, 15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in positive tweets', fontsize=19)
plt.show()

# Sorting Negative Tweets by Polarity
neg_tweets = text_df[text_df.Sentiment == 'Negative']
neg_tweets = neg_tweets.sort_values(['Polarity'], ascending=False)
print(neg_tweets.head())

# show most frequent words in negative tweets
text = ' '.join([word for word in neg_tweets['Tweet']])
plt.figure(figsize=(20, 15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in negative tweets', fontsize=19)
plt.show()

# Sorting neutral Tweets by Polarity
neutral_tweets = text_df[text_df.Sentiment == 'Neutral']
neutral_tweets = neutral_tweets.sort_values(['Polarity'], ascending=False)
print(neutral_tweets.head())

# show most frequent words in neutral tweets
text = ' '.join([word for word in neutral_tweets['Tweet']])
plt.figure(figsize=(20, 15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in neutral tweets', fontsize=19)
plt.show()

tokenized_tweet = text_df.Tweet.apply(lambda x: x.split())
print(tokenized_tweet.info())

model_w2v = gensim.models.Word2Vec(
    tokenized_tweet,
    vector_size=200,  # desired no. of features/independent variables
    window=10,  # context window size
    min_count=2,  # Ignores all words with total frequency lower than 2.
    sg=1,  # 1 for skip-gram model
    hs=0,
    negative=10,  # for negative sampling
    workers=32,  # no.of cores
    seed=34
)

model_w2v.train(tokenized_tweet, total_examples=len(text_df['Tweet']), epochs=20)
#save the model
model_w2v.save("./word2vec_twitterData")

def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += model_w2v.wv[word].reshape((1, size))
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary
            continue
    if count != 0:
        vec /= count
    return vec


wordvec_arrays = np.zeros((len(tokenized_tweet), 200))
for i in range(len(tokenized_tweet)):
    # Attempt to access tokenized tweet at index i
    tweet = tokenized_tweet.iloc[i]
    wordvec_arrays[i, :] = word_vector(tweet, 200)
wordvec_df = pd.DataFrame(wordvec_arrays)
print(wordvec_df.head())

X = wordvec_df


def sentiment_toNum(label):
    if label == "Negative":
        return 2
    elif label == "Positive":
        return 1
    elif label == "Neutral":
        return 0


text_df['Label'] = text_df['Sentiment'].apply(sentiment_toNum)
Y = text_df['Label']

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.2)
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_pred = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc * 100))

# Define the parameter grid
param_grid = {
    'max_depth': [3, 6, 9],
    'n_estimators': [100, 500, 1000],
    'nthread': [3, 6, 9]
}

# Initialize the XGBoost classifier
xgb = XGBClassifier()

# Initialize GridSearchCV
grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='f1_weighted')

# Fit the grid search to the data
grid_search.fit(x_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)


# xgb = XGBClassifier(max_depth=6, n_estimators=1000, nthread=3).fit(x_train, y_train)
# prediction = xgb.predict(x_test)
# # Calculate the F1 score with 'average' parameter set to 'weighted'
# f1 = f1_score(y_test, prediction, average='weighted')
#
# # Print the F1 score
# print(f"F1 Score: {f1}")
# xg_acc = accuracy_score(prediction, y_test)
# print("Test accuracy: {:.2f}%".format(xg_acc * 100))

# vect = CountVectorizer(ngram_range=(1, 2)).fit(text_df['Tweet'])
#
# feature_names = vect.get_feature_names()
# print("Number of features: {}\n".format(len(feature_names)))
# print("First 20 features:\n {}".format(feature_names[:20]))
#
# X = text_df['Tweet']
# Y = text_df['Sentiment']
# X = vect.transform(X)
#
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#
# print("Size of x_train:", (x_train.shape))
# print("Size of y_train:", (y_train.shape))
# print("Size of x_test:", (x_test.shape))
# print("Size of y_test:", (y_test.shape))
#
# import warnings
#
# warnings.filterwarnings('ignore')
#
# logreg = LogisticRegression()
# logreg.fit(x_train, y_train)
# logreg_pred = logreg.predict(x_test)
# logreg_acc = accuracy_score(logreg_pred, y_test)
# print("Test accuracy: {:.2f}%".format(logreg_acc * 100))
#
# print(confusion_matrix(y_test, logreg_pred))
# print("\n")
# print(classification_report(y_test, logreg_pred))
#
# style.use('classic')
# cm = confusion_matrix(y_test, logreg_pred, labels=logreg.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)
# disp.plot()
#
# from sklearn.model_selection import GridSearchCV
#
# param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
# grid = GridSearchCV(LogisticRegression(), param_grid)
# grid.fit(x_train, y_train)
#
# print("Best parameters:", grid.best_params_)
#
# y_pred = grid.predict(x_test)
#
# logreg_acc = accuracy_score(y_pred, y_test)
# print("Test accuracy: {:.2f}%".format(logreg_acc * 100))
#
# print(confusion_matrix(y_test, y_pred))
# print("\n")
# print(classification_report(y_test, y_pred))
#
# from sklearn.svm import LinearSVC
#
# SVCmodel = LinearSVC()
# SVCmodel.fit(x_train, y_train)
#
# svc_pred = SVCmodel.predict(x_test)
# svc_acc = accuracy_score(svc_pred, y_test)
# print("test accuracy: {:.2f}%".format(svc_acc * 100))
#
# print(confusion_matrix(y_test, svc_pred))
# print("\n")
# print(classification_report(y_test, svc_pred))
#
# grid = {
#     'C': [0.01, 0.1, 1, 10],
#     'kernel': ["linear", "poly", "rbf", "sigmoid"],
#     'degree': [1, 3, 5, 7],
#     'gamma': [0.01, 1]
# }
# grid = GridSearchCV(SVCmodel, param_grid)
# grid.fit(x_train, y_train)
#
# print("Best parameter:", grid.best_params_)
#
# y_pred = grid.predict(x_test)
#
# logreg_acc = accuracy_score(y_pred, y_test)
# print("Test accuracy: {:.2f}%".format(logreg_acc * 100))
#
# print(confusion_matrix(y_test, y_pred))
# print("\n")
# print(classification_report(y_test, y_pred))
