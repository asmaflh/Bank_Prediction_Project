import numpy as np
import pandas as pd
import matplotlib

from Sentiment_Analysis.preprocessing import *

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from collections import Counter
import seaborn as sns
from matplotlib import font_manager

eng = pd.read_csv('../Twitter_Datasets/TwitterData_EN.csv')
fr = pd.read_csv('../Twitter_Datasets/TwitterData_FR.csv')
cn = pd.read_csv('../Twitter_Datasets/TwitterData_CN.csv')
de = pd.read_csv('../Twitter_Datasets/TwitterData_DE.csv')
es = pd.read_csv('../Twitter_Datasets/TwitterData_ES.csv')
it = pd.read_csv('../Twitter_Datasets/TwitterData_IT.csv')

print("the number of english tweets are: ", eng['Tweet'].count())
print("the number of french tweets are: ", fr['Tweet'].count())
print("the number of italian tweets are: ", it['Tweet'].count())
print("the number of spanish tweets are: ", es['Tweet'].count())
print("the number of german tweets are: ", de['Tweet'].count())
print("the number of chinese tweets are: ", cn['Tweet'].count())

mylabels = ["English", "French", "German", "Spanish", "Italian", "Chinese"]
y = np.array([
    eng['Tweet'].count(),
    fr['Tweet'].count(),
    it['Tweet'].count(),
    es['Tweet'].count(),
    de['Tweet'].count(),
    cn['Tweet'].count()])

plt.pie(y, labels=mylabels)
plt.title('Distribution of tweets per countries')
plt.show()

# ploting Number of Tweets Mentioning Each Hashtag
datasets = [eng, fr, cn, de, es, it]

hashtags = ['#bank', '#bankcrisis', '#bankcrash', '#BankRun', '#BankingSector',
            '#BankingCollapse', '#BankingRegulations', '#BankingMeltdown', '#TrustworthyBanking', '#BankingExcellence']


def check_hashtags_in_tweet(hashtags, data):
    hashtag_counts = {hashtag: 0 for hashtag in hashtags}
    for hashtag in hashtags:
        contains_column = data['Tweet'].str.contains(hashtag, case=False)
        hashtag_counts[hashtag] = np.sum(contains_column)
    return hashtag_counts


plt.figure(figsize=(18, 6))
for i, dataset in enumerate(datasets):
    hashtag_counts = check_hashtags_in_tweet(hashtags, dataset)
    plt.bar(np.arange(len(hashtags)) + i * 0.1, hashtag_counts.values(), width=0.1, label=mylabels[i])

plt.xlabel('Hashtags')
plt.ylabel('Number of Tweets')
plt.title('Number of Tweets Mentioning Each Hashtag')
plt.xticks(np.arange(len(hashtags)), hashtags, rotation=45)
plt.legend()
plt.show()


def date_to_year(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year


date_to_year(eng)
date_to_year(fr)
date_to_year(es)
date_to_year(it)
date_to_year(de)
date_to_year(cn)

eng['Dataset'] = 'english'
fr['Dataset'] = 'french'
es['Dataset'] = 'spanish'
de['Dataset'] = 'german'
it['Dataset'] = 'italian'
cn['Dataset'] = 'chinese'

# Concatenate the datasets
combined_df = pd.concat([eng, fr, es, de, it, cn])

# Create the countplot
plt.figure(figsize=(8, 8))
sns.countplot(x='Year', hue='Dataset', data=combined_df)
plt.title('Number of Tweets in Each Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='best')
plt.show()

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


def wordBarGraphFunction(df, column, title,fontP):
    topic_words = [z.lower() for y in
                   [x.split() for x in df[column] if isinstance(x, str)]
                   for z in y]
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key=word_count_dict.get, reverse=True)
    popular_words_nonstop = [w for w in popular_words]
    plt.barh(range(20), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:20])])
    plt.yticks([x + 0.1 for x in range(20)], reversed(popular_words_nonstop[0:20]), fontproperties=fontP)
    plt.title(title)
    plt.show()
    print(popular_words_nonstop[0:20])

fontP = [font_manager.FontProperties(fname=r'C:\Users\HP\Downloads\SimHei.ttf', size=14),
         font_manager.FontProperties(fname=r'C:\Windows\Fonts\arial.ttf', size=14)
         ]


plt.figure(figsize=(10, 10))
wordBarGraphFunction(eng, 'Tweet', "Most common words in english dataset",fontP[1])

plt.figure(figsize=(10, 10))
wordBarGraphFunction(fr, 'Tweet', "Most common words in french dataset",fontP[1])

plt.figure(figsize=(10, 10))
wordBarGraphFunction(it, 'Tweet', "Most common words in italian dataset",fontP[1])

plt.figure(figsize=(10, 10))
wordBarGraphFunction(es, 'Tweet', "Most common words in spanish dataset",fontP[1])

plt.figure(figsize=(10, 10))
wordBarGraphFunction(de, 'Tweet', "Most common words in german dataset",fontP[1])

plt.figure(figsize=(10, 10))
wordBarGraphFunction(cn, 'Tweet', "Most common words in chinese dataset",fontP[0])
