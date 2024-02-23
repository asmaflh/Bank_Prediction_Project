import gensim
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib
from xgboost import XGBClassifier

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
from wordcloud import WordCloud
from preprocessing import *

# read csv files
eng = pd.read_csv('../Twitter_Datasets/TwitterData_EN.csv')
fr = pd.read_csv('../Twitter_Datasets/TwitterData_FR.csv')
cn = pd.read_csv('../Twitter_Datasets/TwitterData_CN.csv')
de = pd.read_csv('../Twitter_Datasets/TwitterData_DE.csv')
es = pd.read_csv('../Twitter_Datasets/TwitterData_ES.csv')
it = pd.read_csv('../Twitter_Datasets/TwitterData_IT.csv')

# preprocess of tweets
eng.Tweet = eng['Tweet'].apply(data_processing_en)
fr.Tweet = fr['Tweet'].apply(data_processing_fr)
es.Tweet = es['Tweet'].apply(data_processing_es)
de.Tweet = de['Tweet'].apply(data_processing_de)
it.Tweet = it['Tweet'].apply(data_processing_it)
cn.Tweet = cn['Tweet'].apply(data_processing_cn)

# drop duplicate tweets
eng = eng.drop_duplicates('Tweet')
fr = fr.drop_duplicates('Tweet')
it = it.drop_duplicates('Tweet')
es = es.drop_duplicates('Tweet')
de = de.drop_duplicates('Tweet')
cn = cn.drop_duplicates('Tweet')

# Create a SentimentIntensityAnalyzer object
analyzer = SentimentIntensityAnalyzer()


# Define a function to calculate polarity using VADER
def polarity(text):
    # Use the polarity_scores() method of the analyzer object to get sentiment scores
    sentiment_scores = analyzer.polarity_scores(text)
    # Return the polarity score
    return sentiment_scores['compound']  # 'compound' score represents overall sentiment polarity


eng['Polarity'] = eng['Tweet'].apply(polarity)

print(eng.head(10))


def sentiment(label):
    if label <= -0.05:
        return "Negative"
    elif label >= 0.05:
        return "Positive"
    else:
        return "Neutral"


eng['Sentiment'] = eng['Polarity'].apply(sentiment)

print(eng.head())
# plot the sentiments
fig = plt.figure(figsize=(5, 5))
sns.countplot(x='Sentiment', data=eng)

fig = plt.figure(figsize=(7, 7))
colors = ("yellowgreen", "gold", "red")
wp = {'linewidth': 2, 'edgecolor': "black"}
tags = eng['Sentiment'].value_counts()
explode = (0.1, 0.1, 0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors,
          startangle=90, wedgeprops=wp, explode=explode, label='')

plt.title('Distribution of sentiments')

# Sorting Positive Tweets by Polarity
pos_tweets = eng[eng.Sentiment == 'Positive']
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
neg_tweets = eng[eng.Sentiment == 'Negative']
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
neutral_tweets = eng[eng.Sentiment == 'Neutral']
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




################################################################


def datetime_to_date(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.date


datetime_to_date(eng)



#average polarity
average_polarity = eng.groupby('Date')['Polarity'].mean()

plt.plot(average_polarity.index, average_polarity.values,color="black")
plt.xlabel('Date')
plt.ylabel('Average Polarity')
plt.title('Average Polarity per Date')
plt.show()


#High polarity
high_polarity = eng.groupby('Date')['Polarity'].max()

plt.plot(high_polarity.index, high_polarity.values,color="blue")
plt.xlabel('Date')
plt.ylabel('Maximum Polarity')
plt.title('Maximum Polarity per Date')
plt.show()


#Low polarity
low_polarity = eng.groupby('Date')['Polarity'].min()

plt.plot(low_polarity.index, low_polarity.values)
plt.xlabel('Date')
plt.ylabel('Low Polarity')
plt.title('Low Polarity per Date')
plt.show()

print(average_polarity.values)