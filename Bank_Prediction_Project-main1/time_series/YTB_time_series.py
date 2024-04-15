import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib


matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
from Sentiment_Analysis.preprocessing import *

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# read csv files
eng = pd.read_csv('../Youtube_Datasets/all_comments.csv')



# preprocess of tweets
eng.Text = eng['Text'].apply(data_processing_en)


# drop duplicate tweets
eng = eng.drop_duplicates('Text')

# Convert 'date' column to datetime
eng['Date'] = pd.to_datetime(eng['Date'])

# Sort the DataFrame by the 'date' column
eng = eng.sort_values(by='Date')

# Display the sorted DataFrame
print(eng)




# Create a SentimentIntensityAnalyzer object
analyzer = SentimentIntensityAnalyzer()



# Define a function to calculate polarity using VADER
def polarity(text):
    # Use the polarity_scores() method of the analyzer object to get sentiment scores
    sentiment_scores = analyzer.polarity_scores(text)
    # Return the polarity score
    return sentiment_scores['compound']  # 'compound' score represents overall sentiment polarity


eng['Polarity'] = eng['Text'].apply(polarity)


def sentiment(label):
    if label <= -0.05:
        return "Negative"
    elif label >= 0.05:
        return "Positive"
    else:
        return "Neutral"


eng['Sentiment'] = eng['Polarity'].apply(sentiment)
# Perform seasonal decomposition
result = seasonal_decompose(eng['Polarity'], model='additive', period=365)  # Assuming yearly seasonality

# Plot the decomposition

# Original Series
plt.figure(figsize=(12, 8))
plt.plot(eng['Date'], eng['Polarity'], label='Original', color="blue")
plt.legend(loc='best')
plt.title('Original Series')
plt.show()

# Trend Component
plt.figure(figsize=(12, 8))
plt.plot(eng['Date'], result.trend, label='Trend', color="blue")
plt.legend(loc='best')
plt.title('Trend Component')
plt.show()

# Seasonal Component
plt.figure(figsize=(12, 8))
plt.plot(eng['Date'], result.seasonal, label='Seasonality', color="blue")
plt.legend(loc='best')
plt.title('Seasonal Component')
plt.show()

# Residual Component (Noise)
plt.figure(figsize=(12, 8))
plt.plot(eng['Date'], result.resid, label='Residuals', color="blue")
plt.legend(loc='best')
plt.title('Residual Component')
plt.show()

# Perform Augmented Dickey-Fuller test
result_adf = adfuller(eng['Polarity'])

# Plot the time series
plt.figure(figsize=(12, 8), facecolor='None')
plt.plot(eng['Date'], eng['Polarity'], label='Original', color="blue")
plt.legend(loc='best')
plt.title('Original Series')

# Display the result of the stationarity test
if result_adf[1] <= 0.05:
    stationarity_result = "The series is likely stationary"
else:
    stationarity_result = "The series is likely non-stationary"

# Display the stationarity result as text on the plot
plt.text(eng['Date'].iloc[-1], eng['Polarity'].iloc[-1], stationarity_result,
         verticalalignment='bottom', horizontalalignment='right', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

plt.show()



# Compute and plot ACF
plt.figure(figsize=(12, 6))
plot_acf(eng['Polarity'], lags=50, ax=plt.gca())  # Adjust the number of lags as needed
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()
