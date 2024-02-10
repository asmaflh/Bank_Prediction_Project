import asyncio
from twscrape import API
import pandas as pd

all_tweets = []


async def main():
    api = API()  # or API("path-to.db") - default is `accounts.db`

    # ADD ACCOUNTS (for CLI usage see BELOW)
    # await api.pool.add_account("user", "pass1", "email@gmail.com", "pass")
    # await api.pool.login_all()
    queries = [
        "(#bank) lang:en until:2024-01-01 since:2022-01-01",
        "(#bankcrisis) lang:en until:2024-01-01 since:2022-01-01",
        "(#bankcrash) lang:en until:2024-01-01 since:2022-01-01",
        "(#BankRun) lang:en until:2024-01-01 since:2022-01-01",
        "(#BankingSector) lang:en until:2024-01-01 since:2022-01-01",
        "(#BankingCollapse) lang:en until:2024-01-01 since:2022-01-01",
        "(#BankingRegulations) lang:en until:2024-01-01 since:2022-01-01",
        "(#BankingMeltdown) lang:en until:2024-01-01 since:2022-01-01",
        "(#TrustworthyBanking) lang:en until:2024-01-01 since:2022-01-01",
        "(#BankingExcellence) lang:en until:2024-01-01 since:2022-01-01",
    ]
    for q in queries:
        async for tweet in api.search(q, limit=5000):
            all_tweets.append(
                [tweet.id, tweet.user.id, tweet.user.username, tweet.date, tweet.user.location, tweet.rawContent,
                 tweet.media, tweet.likeCount, tweet.retweetCount, tweet.replyCount])

    print("finishing....")
    df = pd.DataFrame(all_tweets,
                      columns=['TweetId', 'UserId', 'UserUsername', 'Date', 'Location', 'Tweet', 'Media', 'Likes',
                               'Retweets', 'Comments'])
    print(df)
    df.to_csv('TwitterData.csv')


if __name__ == "__main__":
    asyncio.run(main())
