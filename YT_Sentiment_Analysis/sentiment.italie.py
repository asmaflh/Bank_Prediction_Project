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

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# read csv files
it = pd.read_csv('../Youtube_Datasets/all_commentsIT.csv')

# preprocess of tweets
it.Text = it['Text'].apply(data_processing_it)

# drop duplicate tweets
it = it.drop_duplicates('Text')

# Convert 'date' column to datetime
it['Date'] = pd.to_datetime(it['Date'])

# Sort the DataFrame by the 'date' column
it = it.sort_values(by='Date')

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

it.to_csv('../Sentiment_Youtube_Data/it_comm.csv')



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
pos_texts = it[it.Sentiment == 'positive']
pos_texts = pos_texts.sort_values(['Polarity'], ascending=False)
print(pos_texts['Text'].head(10))

# show most frequent words in positive tweets
text = ' '.join([word for word in pos_texts['Text']])
plt.figure(figsize=(20, 15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in positive comments', fontsize=19)
plt.show()

# Sorting Negative Tweets by Polarity
neg_texts = it[it.Sentiment == 'negative']
neg_texts = neg_texts.sort_values(['Polarity'], ascending=False)
print(neg_texts.head())

# show most frequent words in negative tweets
text = ' '.join([word for word in neg_texts['Text']])
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


