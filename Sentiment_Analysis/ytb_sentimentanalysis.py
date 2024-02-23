import pandas as pd
import re
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style
from nltk import PorterStemmer

style.use('ggplot')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from cleantext import clean

# read csv file
df = pd.read_csv('all_comments.csv')


# display infos about the dataset
print(df.head())
print(df.info())
print(df.isnull().sum)
print(df.columns)


# extract tweet column
text_df = df.drop(['CommentID','UserName','Date'], axis=1)

print(text_df['Text'].iloc[0], "\n")
print(text_df['Text'].iloc[1], "\n")
print(text_df['Text'].iloc[2], "\n")
print(text_df['Text'].iloc[3], "\n")
print(text_df['Text'].iloc[4], "\n")

print(text_df.info())


# clean the dataset
def data_processing(text):
    text = text.lower()
    # remove urls
    text = re.sub(r"https\S+|www\S+https\S+", '', text, flags=re.MULTILINE)
    #remove special caracters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # remove numbers
    text = re.sub(r'\d+', '', text)
    # Removing User Mentions and Hashtags
    text = re.sub(r'@\w+|\#\w+', '', text)
    # Removing Punctuation:
    text = re.sub(r'[^\w\s]', '', text)
    text=clean(text, no_emoji=True)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if len(w) >= 4 and not w in stop_words]
    stemmer = PorterStemmer()
    stemmer_text = [stemmer.stem(word) for word in filtered_text]
    return " ".join(stemmer_text)


text_df.Text= text_df['Text'].apply(data_processing)
print(text_df.head())
print(text_df['Text'].iloc[0], "\n")
print(text_df['Text'].iloc[1], "\n")
print(text_df['Text'].iloc[2], "\n")
print(text_df['Text'].iloc[3], "\n")
print(text_df['Text'].iloc[4], "\n")
# drop duplicate tweets
text_df = text_df.drop_duplicates('Text')

# Create a SentimentIntensityAnalyzer object
analyzer = SentimentIntensityAnalyzer()


# Define a function to calculate polarity using VADER
def polarity(text):
    # Use the polarity_scores() method of the analyzer object to get sentiment scores
    sentiment_scores = analyzer.polarity_scores(text)
    # Return the polarity score
    return sentiment_scores['compound']  # 'compound' score represents overall sentiment polarity


text_df['Polarity'] = text_df['Text'].apply(polarity)

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

# Sorting Positive comments by Polarity
pos_comments = text_df[text_df.Sentiment == 'Positive']
pos_comments = pos_comments.sort_values(['Polarity'], ascending=False)
print(pos_comments['Text'].head(10))

# show most frequent words in positive comments
text = ' '.join([word for word in pos_comments['Text']])
plt.figure(figsize=(20, 15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in positive comments', fontsize=19)
plt.show()

# Sorting Negative comments by Polarity
neg_comments = text_df[text_df.Sentiment == 'Negative']
neg_comments = neg_comments.sort_values(['Polarity'], ascending=False)
print(neg_comments.head())

# show most frequent words in negative comments
text = ' '.join([word for word in neg_comments['Text']])
plt.figure(figsize=(20, 15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in negative comments', fontsize=19)
plt.show()

# Sorting neutral comments by Polarity
neutral_comments = text_df[text_df.Sentiment == 'Neutral']
neutral_comments = neutral_comments.sort_values(['Polarity'], ascending=False)
print(neutral_comments.head())

# show most frequent words in neutral comments
text = ' '.join([word for word in neutral_comments['Text']])
plt.figure(figsize=(20, 15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in neutral comments', fontsize=19)
plt.show()


