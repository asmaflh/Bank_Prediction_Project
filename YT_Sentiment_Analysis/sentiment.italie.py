
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style
from transformers import pipeline

style.use('ggplot')
from wordcloud import WordCloud
from Sentiment_Analysis.preprocessing import *

# read csv files
it = pd.read_csv('../Youtube_Datasets/all_commentsIT.csv')

# preprocess of tweets
it.Text = it['Text'].apply(data_processing_it)

# drop duplicate tweets
it = it.drop_duplicates('Text')

classifier = pipeline("text-classification", model='MilaNLProc/feel-it-italian-sentiment', top_k=2)


def sentiment(text):
    try:
        prediction = classifier(text)[0][0]['label']
    except Exception:
        prediction = "negative"
    return prediction


it['Sentiment'] = it['Text'].apply(sentiment)

print(it.head(10))


def polarity(label):
    if label == "negative":
        return -1
    else:
        return 1


it['Polarity'] = it['Sentiment'].apply(polarity)

print(it.head())
# plot the sentiments
fig = plt.figure(figsize=(5, 5))
sns.countplot(x='Sentiment', data=it)

fig = plt.figure(figsize=(7, 7))
colors = ("yellowgreen", "gold", "red")
wp = {'linewidth': 2, 'edgecolor': "black"}
tags = it['Sentiment'].value_counts()
explode = (0.1, 0)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors,
          startangle=90, wedgeprops=wp, explode=explode, label='')

plt.title('Distribution of sentiments')

# Sorting Positive Tweets by Polarity
pos_tweets = it[it.Sentiment == 'positive']
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
neg_tweets = it[it.Sentiment == 'negative']
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



################################################################


def datetime_to_date(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.date


datetime_to_date(it)

# average polarity
average_polarity = it.groupby('Date')['Polarity'].mean()

plt.plot(average_polarity.index, average_polarity.values, color="black")
plt.xlabel('Date')
plt.ylabel('Average Polarity')
plt.title('Average Polarity per Date')
plt.show()

# High polarity
high_polarity = it.groupby('Date')['Polarity'].max()

plt.plot(high_polarity.index, high_polarity.values, color="blue")
plt.xlabel('Date')
plt.ylabel('Maximum Polarity')
plt.title('Maximum Polarity per Date')
plt.show()

# Low polarity
low_polarity = it.groupby('Date')['Polarity'].min()

plt.plot(low_polarity.index, low_polarity.values)
plt.xlabel('Date')
plt.ylabel('Low Polarity')
plt.title('Low Polarity per Date')
plt.show()

print(average_polarity.values)
