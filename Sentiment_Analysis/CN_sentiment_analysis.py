import pandas as pd
import seaborn as sns
from bixin import predict
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
from wordcloud import WordCloud
from preprocessing import *

# read csv files
cn = pd.read_csv('../Twitter_Datasets/TwitterData_CN.csv')


# preprocess of tweets

cn.Tweet = cn['Tweet'].apply(data_processing_cn)

# drop duplicate tweets
cn = cn.drop_duplicates('Tweet')
def polarity(text):
    try:
        pol = predict(text)

    except Exception:
        pol = 0
    return pol


cn['Polarity'] = cn['Tweet'].apply(polarity)

print(cn.head(10))


def sentiment(label):
    if label <0:
        return "Negative"
    elif label >0:
        return "Positive"
    else:
        return "Neutral"


cn['Sentiment'] = cn['Polarity'].apply(sentiment)
# cn.to_csv("TwitterData_Class_CN.csv")

print(cn.head())
# plot the sentiments
fig = plt.figure(figsize=(5, 5))
sns.countplot(x='Sentiment', data=cn)

fig = plt.figure(figsize=(7, 7))
colors = ("yellowgreen", "gold", "red")
wp = {'linewidth': 2, 'edgecolor': "black"}
tags = cn['Sentiment'].value_counts()
explode = (0.1, 0.1, 0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors,
          startangle=90, wedgeprops=wp, explode=explode, label='')

plt.title('Distribution of sentiments')

# Sorting Positive Tweets by Polarity
pos_tweets = cn[cn.Sentiment == 'Positive']
pos_tweets = pos_tweets.sort_values(['Polarity'], ascending=False)
print(pos_tweets['Tweet'].head(10))

font_p=r'C:\Users\HP\Downloads\SimHei.ttf'
# show most frequent words in positive tweets
text = ' '.join([word for word in pos_tweets['Tweet']])
plt.figure(figsize=(20, 15), facecolor='None')
wordcloud = WordCloud(font_path = font_p,max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in positive tweets', fontsize=19)
plt.show()

# Sorting Negative Tweets by Polarity
neg_tweets = cn[cn.Sentiment == 'Negative']
neg_tweets = neg_tweets.sort_values(['Polarity'], ascending=False)
print(neg_tweets.head())

# show most frequent words in negative tweets
text = ' '.join([word for word in neg_tweets['Tweet']])
plt.figure(figsize=(20, 15), facecolor='None')
wordcloud = WordCloud(font_path = font_p,max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in negative tweets', fontsize=19)
plt.show()

# Sorting neutral Tweets by Polarity
neutral_tweets = cn[cn.Sentiment == 'Neutral']
neutral_tweets = neutral_tweets.sort_values(['Polarity'], ascending=False)
print(neutral_tweets.head())

# show most frequent words in neutral tweets
text = ' '.join([word for word in neutral_tweets['Tweet']])
plt.figure(figsize=(20, 15), facecolor='None')
wordcloud = WordCloud(font_path = font_p,max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in neutral tweets', fontsize=19)
plt.show()




################################################################


def datetime_to_date(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.date


datetime_to_date(cn)



#average polarity
average_polarity = cn.groupby('Date')['Polarity'].mean()

plt.plot(average_polarity.index, average_polarity.values,color="black")
plt.xlabel('Date')
plt.ylabel('Average Polarity')
plt.title('Average Polarity per Date')
plt.show()


#High polarity
high_polarity = cn.groupby('Date')['Polarity'].max()

plt.plot(high_polarity.index, high_polarity.values,color="blue")
plt.xlabel('Date')
plt.ylabel('Maximum Polarity')
plt.title('Maximum Polarity per Date')
plt.show()


#Low polarity
low_polarity = cn.groupby('Date')['Polarity'].min()

plt.plot(low_polarity.index, low_polarity.values)
plt.xlabel('Date')
plt.ylabel('Low Polarity')
plt.title('Low Polarity per Date')
plt.show()

print(average_polarity.values)


