import pandas as pd
import seaborn as sns
from bixin import predict
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
from wordcloud import WordCloud
from Sentiment_Analysis.preprocessing import *

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# read csv files
cn = pd.read_csv('../Youtube_Datasets/all_commentsCH.csv')


# preprocess of tweets

cn.Text = cn['Text'].apply(data_processing_cn)

# drop duplicate tweets
cn = cn.drop_duplicates('Text')

# Convert 'date' column to datetime
cn['Date'] = pd.to_datetime(cn['Date'])

# Sort the DataFrame by the 'date' column
cn = cn.sort_values(by='Date')


def polarity(text):
    try:
        pol = predict(text)

    except Exception:
        pol = 0
    return pol


cn['Polarity'] = cn['Text'].apply(polarity)

print(cn.head(10))


def sentiment(label):
    if label <0:
        return "Negative"
    elif label >0:
        return "Positive"
    else:
        return "Neutral"


cn['Sentiment'] = cn['Polarity'].apply(sentiment)

# cn.to_csv('../Sentiment_Youtube_Data/ch_comm.csv')


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

# Sorting Positive comments by Polarity
pos_texts = cn[cn.Sentiment == 'Positive']
pos_texts = pos_texts.sort_values(['Polarity'], ascending=False)
print(pos_texts['Text'].head(10))

font_p=r'C:\Users\DELL\Desktop\3SC\SimHei.ttf'
# show most frequent words in positive tweets
text = ' '.join([word for word in pos_texts['Text']])
plt.figure(figsize=(20, 15), facecolor='None')
wordcloud = WordCloud(font_path = font_p,max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in positive commets', fontsize=19)
plt.show()

# Sorting Negative comments  by Polarity
neg_texts = cn[cn.Sentiment == 'Negative']
neg_texts = neg_texts.sort_values(['Polarity'], ascending=False)
print(neg_texts.head())

# show most frequent words in negative tweets
text = ' '.join([word for word in neg_texts['Text']])
plt.figure(figsize=(20, 15), facecolor='None')
wordcloud = WordCloud(font_path = font_p,max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in negative comments', fontsize=19)
plt.show()

# Sorting neutral Tweets by Polarity
neutral_texts = cn[cn.Sentiment == 'Neutral']
neutral_texts = neutral_texts.sort_values(['Polarity'], ascending=False)
print(neutral_texts.head())

# show most frequent words in neutral tweets
text = ' '.join([word for word in neutral_texts['Text']])
plt.figure(figsize=(20, 15), facecolor='None')
wordcloud = WordCloud(font_path = font_p,max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in neutral comments', fontsize=19)
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
