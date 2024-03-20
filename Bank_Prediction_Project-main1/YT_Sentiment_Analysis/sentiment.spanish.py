import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
from wordcloud import WordCloud
from Sentiment_Analysis.preprocessing import *
from pysentimiento import create_analyzer

# read csv files
es = pd.read_csv('../Youtube_Datasets/all_commentsES.csv')

# preprocess of tweets
es.Tweet = es['Text'].apply(data_processing_es)

# drop duplicate tweets
es = es.drop_duplicates('Text')

###############


analyzer = create_analyzer(task="sentiment", lang="es")


def sentiment(text):
    try:
        sen = analyzer.predict(text).output
    except Exception:
        sen = "NEU"
    if sen == "NEG":
        sen= "Negative"
    elif sen == "POS":
        sen="Positive"
    else:
        sen="Neutral"
    return sen


es['Sentiment'] = es['Text'].apply(sentiment)

print(es.head(10))


def polarity(label):
    if label == "Negative":
        return -1
    elif label == "Positive":
        return 1
    else:
        return 0


es['Polarity'] = es['Sentiment'].apply(polarity)

print(es.head())
# plot the sentiments
fig = plt.figure(figsize=(5, 5))
sns.countplot(x='Sentiment', data=es)

fig = plt.figure(figsize=(7, 7))
colors = ("yellowgreen", "gold", "red")
wp = {'linewidth': 2, 'edgecolor': "black"}
tags = es['Sentiment'].value_counts()
explode = (0.1, 0.1, 0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors,
          startangle=90, wedgeprops=wp, explode=explode, label='')

plt.title('Distribution of sentiments')

# Sorting Positive Tweets by Polarity
pos_tweets = es[es.Sentiment == 'Positive']
pos_tweets = pos_tweets.sort_values(['Polarity'], ascending=False)
print(pos_tweets['Text'].head(10))

# show most frequent words in positive tweets
text = ' '.join([word for word in pos_tweets['Text']])
plt.figure(figsize=(20, 15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in positive comments', fontsize=19)
plt.show()

# Sorting Negative Tweets by Polarity
neg_tweets = es[es.Sentiment == 'Negative']
neg_tweets = neg_tweets.sort_values(['Polarity'], ascending=False)
print(neg_tweets.head())

# show most frequent words in negative tweets
text = ' '.join([word for word in neg_tweets['Text']])
plt.figure(figsize=(20, 15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in negative comments', fontsize=19)
plt.show()

# Sorting neutral Tweets by Polarity
neutral_tweets = es[es.Sentiment == 'Neutral']
neutral_tweets = neutral_tweets.sort_values(['Polarity'], ascending=False)
print(neutral_tweets.head())

# show most frequent words in neutral tweets
text = ' '.join([word for word in neutral_tweets['Text']])
plt.figure(figsize=(20, 15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in neutral comments', fontsize=19)
plt.show()


################################################################


def datetime_to_date(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.date


datetime_to_date(es)

# average polarity
average_polarity = es.groupby('Date')['Polarity'].mean()

plt.plot(average_polarity.index, average_polarity.values, color="black")
plt.xlabel('Date')
plt.ylabel('Average Polarity')
plt.title('Average Polarity per Date')
plt.show()

# High polarity
high_polarity = es.groupby('Date')['Polarity'].max()

plt.plot(high_polarity.index, high_polarity.values, color="blue")
plt.xlabel('Date')
plt.ylabel('Maximum Polarity')
plt.title('Maximum Polarity per Date')
plt.show()

# Low polarity
low_polarity = es.groupby('Date')['Polarity'].min()

plt.plot(low_polarity.index, low_polarity.values)
plt.xlabel('Date')
plt.ylabel('Low Polarity')
plt.title('Low Polarity per Date')
plt.show()

print(average_polarity.values)
