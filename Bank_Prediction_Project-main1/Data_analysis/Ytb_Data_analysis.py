import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from Sentiment_Analysis.preprocessing import *
from collections import Counter
import seaborn as sns

# read csv files
eng = pd.read_csv('../Youtube_Datasets/all_comments.csv')
fr = pd.read_csv('../Youtube_Datasets/all_commentsFR.csv')
cn = pd.read_csv('../Youtube_Datasets/all_commentsCH.csv')
de = pd.read_csv('../Youtube_Datasets/all_commentsALL.csv')
es = pd.read_csv('../Youtube_Datasets/all_commentsES.csv')
it = pd.read_csv('../Youtube_Datasets/all_commentsIT.csv')


print("the number of english comments are: ", eng['Text'].count())
print("the number of french comments are: ", fr['Text'].count())
print("the number of italian comments are: ", it['Text'].count())
print("the number of spanish comments are: ", es['Text'].count())
print("the number of german comments are: ", de['Text'].count())
print("the number of chinese comments are: ", cn['Text'].count())

mylabels = ["English", "French", "German", "Spanish", "Italian", "Chinese"]
y = np.array([
    eng['Text'].count(),
    fr['Text'].count(),
    it['Text'].count(),
    es['Text'].count(),
    de['Text'].count(),
    cn['Text'].count()])

plt.pie(y, labels=mylabels)
plt.title('Distribution of comments per countries')
plt.show()

# ploting Number of Tweets Mentioning Each Hashtag
datasets = [eng, fr, cn, de, es, it]



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
plt.title('Number of Comments in Each Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='best')
plt.show()

# preprocess of tweets
eng.Text = eng['Text'].apply(data_processing_en)
fr.Text = fr['Text'].apply(data_processing_fr)
es.Text = es['Text'].apply(data_processing_es)
de.Text = de['Text'].apply(data_processing_de)
it.Text = it['Text'].apply(data_processing_it)
cn.Text = cn['Text'].apply(data_processing_cn)

# drop duplicate tweets
eng = eng.drop_duplicates('Text')
fr = fr.drop_duplicates('Text')
it = it.drop_duplicates('Text')
es = es.drop_duplicates('Text')
de = de.drop_duplicates('Text')
cn = cn.drop_duplicates('Text')


def wordBarGraphFunction(df, column, title):
    topic_words = [z.lower() for y in
                   [x.split() for x in df[column] if isinstance(x, str)]
                   for z in y]
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key=word_count_dict.get, reverse=True)
    popular_words_nonstop = [w for w in popular_words]
    plt.barh(range(20), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:20])])
    plt.yticks([x + 0.1 for x in range(20)], reversed(popular_words_nonstop[0:20]))
    plt.title(title)
    plt.show()


plt.figure(figsize=(10, 10))
wordBarGraphFunction(eng, 'Text', "Most common words in english dataset")

plt.figure(figsize=(10, 10))
wordBarGraphFunction(fr, 'Text', "Most common words in french dataset")

plt.figure(figsize=(10, 10))
wordBarGraphFunction(it, 'Text', "Most common words in italian dataset")

plt.figure(figsize=(10, 10))
wordBarGraphFunction(es, 'Text', "Most common words in spanish dataset")

plt.figure(figsize=(10, 10))
wordBarGraphFunction(de, 'Text', "Most common words in german dataset")

# plt.figure(figsize=(10, 10))
# wordBarGraphFunction(text_cn, 'Tweet', "Most common words in chinese dataset")


