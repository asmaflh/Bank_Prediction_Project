import pandas as pd
import seaborn as sns
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer
import matplotlib

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
from wordcloud import WordCloud
from Sentiment_Analysis.preprocessing import *

# read csv files
fr = pd.read_csv('../Youtube_Datasets/all_commentsFR.csv')

# preprocess of tweets
fr.Text = fr['Text'].apply(data_processing_fr)

# drop duplicate tweets
fr = fr.drop_duplicates('Text')

# Convert 'date' column to datetime
fr['Date'] = pd.to_datetime(fr['Date'])

# Sort the DataFrame by the 'date' column
fr = fr.sort_values(by='Date')


tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())


def polarity(text):
    try:
        blob1 = tb(text)
        pol, sub = blob1.sentiment
    except Exception:
        pol = 0
    return pol


fr['Polarity'] = fr['Text'].apply(polarity)

print(fr.head(10))


def sentiment(label):
    if label < 0:
        return "Negative"
    elif label > 0:
        return "Positive"
    else:
        return "Neutral"


fr['Sentiment'] = fr['Polarity'].apply(sentiment)

print(fr.head())
# # plot the sentiments
# fig = plt.figure(figsize=(5, 5))
# sns.countplot(x='Sentiment', data=fr)
#
# fig = plt.figure(figsize=(7, 7))
# colors = ("yellowgreen", "gold", "red")
# wp = {'linewidth': 2, 'edgecolor': "black"}
# tags = fr['Sentiment'].value_counts()
# explode = (0.1, 0.1, 0.1)
# tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors,
#           startangle=90, wedgeprops=wp, explode=explode, label='')
#
# plt.title('Distribution of sentiments')
#
# # Sorting Positive Tweets by Polarity
# pos_texts = fr[fr.Sentiment == 'Positive']
# pos_texts = pos_texts.sort_values(['Polarity'], ascending=False)
# print(pos_texts['Text'].head(10))
#
# # show most frequent words in positive tweets
# text = ' '.join([word for word in pos_texts['Text']])
# plt.figure(figsize=(20, 15), facecolor='None')
# wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.title('Most frequent words in positive comments', fontsize=19)
# plt.show()
#
# # Sorting Negative Tweets by Polarity
# neg_texts = fr[fr.Sentiment == 'Negative']
# neg_texts = neg_texts.sort_values(['Polarity'], ascending=False)
# print(neg_texts.head())
#
# # show most frequent words in negative tweets
# text = ' '.join([word for word in neg_texts['Text']])
# plt.figure(figsize=(20, 15), facecolor='None')
# wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.title('Most frequent words in negative comments', fontsize=19)
# plt.show()
#
# # Sorting neutral Tweets by Polarity
# neutral_texts = fr[fr.Sentiment == 'Neutral']
# neutral_texts = neutral_texts.sort_values(['Polarity'], ascending=False)
# print(neutral_texts.head())
#
# # show most frequent words in neutral tweets
# text = ' '.join([word for word in neutral_texts['Text']])
# plt.figure(figsize=(20, 15), facecolor='None')
# wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.title('Most frequent words in neutral comments', fontsize=19)
# plt.show()
#
#
#
#
# ################################################################
#
#
# def datetime_to_date(df):
#     df['Date'] = pd.to_datetime(df['Date'])
#     df['Date'] = df['Date'].dt.date
#
#
# datetime_to_date(fr)
#
#
#
# #average polarity
# average_polarity = fr.groupby('Date')['Polarity'].mean()
#
# plt.plot(average_polarity.index, average_polarity.values,color="black")
# plt.xlabel('Date')
# plt.ylabel('Average Polarity')
# plt.title('Average Polarity per Date')
# plt.show()
#
#
# #High polarity
# high_polarity = fr.groupby('Date')['Polarity'].max()
#
# plt.plot(high_polarity.index, high_polarity.values,color="blue")
# plt.xlabel('Date')
# plt.ylabel('Maximum Polarity')
# plt.title('Maximum Polarity per Date')
# plt.show()
#
#
# #Low polarity
# low_polarity = fr.groupby('Date')['Polarity'].min()
#
# plt.plot(low_polarity.index, low_polarity.values)
# plt.xlabel('Date')
# plt.ylabel('Low Polarity')
# plt.title('Low Polarity per Date')
# plt.show()
#
# print(average_polarity.values)

############################################################################

# Perform seasonal decomposition
result = seasonal_decompose(fr['Polarity'], model='additive', period=365)  # Assuming yearly seasonality

# Plot the decomposition

# Original Series
plt.figure(figsize=(12, 8))
plt.plot(fr['Date'], fr['Polarity'], label='Original', color="blue")
plt.legend(loc='best')
plt.title('Original Series')
plt.show()

# Trend Component
plt.figure(figsize=(12, 8))
plt.plot(fr['Date'], result.trend, label='Trend', color="blue")
plt.legend(loc='best')
plt.title('Trend Component')
plt.show()

# Seasonal Component
plt.figure(figsize=(12, 8))
plt.plot(fr['Date'], result.seasonal, label='Seasonality', color="blue")
plt.legend(loc='best')
plt.title('Seasonal Component')
plt.show()

# Residual Component (Noise)
plt.figure(figsize=(12, 8))
plt.plot(fr['Date'], result.resid, label='Residuals', color="blue")
plt.legend(loc='best')
plt.title('Residual Component')
plt.show()

# Perform Augmented Dickey-Fuller test
result_adf = adfuller(fr['Polarity'])

# Plot the time series
plt.figure(figsize=(12, 8), facecolor='None')
plt.plot(fr['Date'], fr['Polarity'], label='Original', color="blue")
plt.legend(loc='best')
plt.title('Original Series')

# Display the result of the stationarity test
if result_adf[1] <= 0.05:
    stationarity_result = "The series is likely stationary"
else:
    stationarity_result = "The series is likely non-stationary"

# Display the stationarity result as text on the plot
plt.text(fr['Date'].iloc[-1], fr['Polarity'].iloc[-1], stationarity_result,
         verticalalignment='bottom', horizontalalignment='right', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

plt.show()



# Compute and plot ACF
plt.figure(figsize=(12, 6))
plot_acf(fr['Polarity'], lags=50, ax=plt.gca())  # Adjust the number of lags as needed
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()

