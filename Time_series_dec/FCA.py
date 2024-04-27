import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
from statsmodels.tsa.seasonal import seasonal_decompose

# read csv files
eng = pd.read_csv('../Sentiment_Twitter_Data/TwitterData_Class_EN.csv')
# Convert 'date' column to datetime
eng['Date'] = pd.to_datetime(eng['Date'])
# Sort the DataFrame by the 'date' column
eng = eng.sort_values(by='Date')

# Resample the data on a weekly basis and calculate mean polarity for each week
weekly_polarity = eng.resample('W-Mon', on='Date')['Polarity'].mean()
weekly_polarity.fillna(0, inplace=True)

# Plot the weekly polarity
plt.figure(figsize=(12, 6))
weekly_polarity.plot(marker='o', color='b')
plt.title('Weekly Polarity')
plt.xlabel('Weeks')
plt.ylabel('Average Polarity')
plt.grid(True)
plt.show()

# Create a new DataFrame containing week number and mean polarity
weekly_polarity_dataset = pd.DataFrame({
    'Week_Number': weekly_polarity.index,  # week number
    'Mean_Polarity': weekly_polarity.values  # Use the mean polarity values
})
# Convert 'date' column to datetime
weekly_polarity_dataset['Week_Number'] = pd.to_datetime(weekly_polarity_dataset['Week_Number'])

# Sort the DataFrame by the 'date' column
weekly_polarity_dataset = weekly_polarity_dataset.sort_values(by='Week_Number')

print(weekly_polarity_dataset)

# Perform seasonal decomposition
result = seasonal_decompose(weekly_polarity_dataset['Mean_Polarity'], model='additive', period=52)

# Trend Component
plt.figure(figsize=(12, 8))
plt.plot(weekly_polarity_dataset['Week_Number'], result.trend, label='Trend', color="blue")
plt.legend(loc='best')
plt.xlabel('Weeks')
plt.ylabel('Average Polarity')
plt.title('Trend Component')
plt.show()

# Residual Component (Noise)
plt.figure(figsize=(12, 8))
plt.plot(weekly_polarity_dataset['Week_Number'], result.resid, label='Residuals', color="blue")
plt.legend(loc='best')
plt.xlabel('Weeks')
plt.ylabel('Average Polarity')
plt.title('Residual Component')
plt.show()

weekly_polarity_dataset['Polarity_Trend'] = result.trend.values

############################

# read csv files
dji = pd.read_csv('DJI.csv')
dji['Date'] = pd.to_datetime(dji['Date'])
dji = dji.sort_values(by='Date')

print(dji.head(10))
# dji dataset Plotting
figure, axis = plt.subplots(2, 3)
X = dji['Date']

axis[0, 0].plot(X, dji['Open'])
axis[0, 0].set_title("Open Price")

axis[0, 1].plot(X, dji['High'])
axis[0, 1].set_title("High Price")

axis[1, 0].plot(X, dji['Low'])
axis[1, 0].set_title("Low Price")

axis[1, 1].plot(X, dji['Close'])
axis[1, 1].set_title("Close Price")

axis[0, 2].plot(X, dji['Adj Close'])
axis[0, 2].set_title("Adj Close Price")

axis[1, 2].plot(X, dji['Volume'])
axis[1, 2].set_title("Volume")
# Combine all the operations and display
plt.show()

weekly_Close = dji.resample('W-Mon', on='Date')['Close'].mean()
weekly_Close.fillna(0, inplace=True)

plt.figure(figsize=(12, 6))
weekly_Close.plot(marker='o', color='b')
plt.title('Weekly Close Price')
plt.xlabel('Weeks')
plt.ylabel('Average Close Price')
plt.grid(True)
plt.show()

weekly_Close_dataset = pd.DataFrame({
    'Week_Number': weekly_Close.index,
    'Mean_Close': weekly_Close.values
})
weekly_Close_dataset['Week_Number'] = pd.to_datetime(weekly_Close_dataset['Week_Number'])

weekly_Close_dataset = weekly_Close_dataset.sort_values(by='Week_Number')

print(weekly_Close_dataset)

# Perform seasonal decomposition
close_result = seasonal_decompose(weekly_Close_dataset['Mean_Close'], model='additive', period=52)

# Trend Component
plt.figure(figsize=(12, 6))
plt.plot(weekly_Close_dataset['Week_Number'], close_result.trend, label='Trend', color="blue")
plt.legend(loc='best')
plt.xlabel('Weeks')
plt.ylabel('Average Close Price')
plt.title('Trend Component')
plt.show()

# Residual Component (Noise)
plt.figure(figsize=(12, 6))
plt.plot(weekly_Close_dataset['Week_Number'], close_result.resid, label='Residuals', color="blue")
plt.legend(loc='best')
plt.xlabel('Weeks')
plt.ylabel('Average Close Price')
plt.title('Residual Component')
plt.show()
print(close_result.trend)
# add the trend as column
weekly_Close_dataset['Trend'] = close_result.trend.values

print(weekly_Close_dataset.head(10))

# merge the two datasets
weekly_polarity_dataset['Week_Number'] = weekly_polarity_dataset['Week_Number'].dt.tz_localize(None)
weekly_Close_dataset['Week_Number'] = weekly_Close_dataset['Week_Number'].dt.tz_localize(None)
merged_df = pd.merge(weekly_polarity_dataset, weekly_Close_dataset, on='Week_Number', how='right')
#############
# plot in same plotting
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Weeks')
ax1.set_ylabel('Polarity Trend', color=color)
ax1.plot(merged_df['Polarity_Trend'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:green'
ax2.set_ylabel('Close trend', color=color)
ax2.plot(merged_df['Trend'], color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.suptitle('comparision between polarity trend and close trend', fontweight="bold")
plt.show()

#############
merged_df['Week_Number'] = merged_df['Week_Number'].dt.strftime('%Y-%m-%d')
merged_df['Week_Number'] = 'Week_' + (merged_df.index + 1).astype(str)
merged_df.set_index('Week_Number', inplace=True)
print(merged_df.index)
print(merged_df.head(10))


def intervals(column):
    interval_size = 50  # Define the size of each interval
    num_intervals = len(merged_df) // interval_size

    # Determine if the curve is increasing, decreasing, or stationary in each interval
    results = []  # List to store results for each interval
    for i in range(num_intervals):
        start_week = merged_df.index[i * interval_size]
        end_week = merged_df.index[min((i + 1) * interval_size - 1, len(merged_df) - 1)]
        interval_slope = (merged_df[column].iloc[(i + 1) * interval_size - 1] - merged_df[column].iloc[
            i * interval_size]) / interval_size
        trend = "Increasing" if interval_slope > 0 else "Decreasing" if interval_slope < 0 else "Stationary"
        results.append((start_week, end_week, trend))

    return results


# Apply the intervals function to merged_df
trend_results = intervals('Polarity_Trend')

# Create three new columns based on the result
merged_df['Sen_I'] = 0
merged_df['Sen_D'] = 0
merged_df['Sen_S'] = 0

for start_week, end_week, trend in trend_results:
    if trend == 'Increasing':
        merged_df.loc[start_week:end_week, 'Sen_I'] = 1
    elif trend == 'Decreasing':
        merged_df.loc[start_week:end_week, 'Sen_D'] = 1
    else:
        merged_df.loc[start_week:end_week, 'Sen_S'] = 1
########################

# Apply the intervals function to merged_df
trend_resultss = intervals('Trend')

# Create three new columns based on the result
merged_df['Close_I'] = 0
merged_df['Close_D'] = 0
merged_df['Close_S'] = 0

for start_week, end_week, trend in trend_resultss:
    if trend == 'Increasing':
        merged_df.loc[start_week:end_week, 'Close_I'] = 1
    elif trend == 'Decreasing':
        merged_df.loc[start_week:end_week, 'Close_D'] = 1
    else:
        merged_df.loc[start_week:end_week, 'Close_S'] = 1
merged_df = merged_df.drop(['Mean_Polarity', 'Polarity_Trend', 'Mean_Close', 'Trend'], axis=1)
print(merged_df.head(50))
merged_df.to_csv('first_context.csv')
from concepts import Context

bools_df = merged_df[['Sen_I', 'Sen_D', 'Sen_S', 'Close_I', 'Close_D', 'Close_S']]
bools = bools_df.values.tolist()

c = Context(objects=merged_df.index, properties=merged_df.columns, bools=bools)

print(c.objects)
print(c.properties)
print(c.bools)

for extent, intent in c.lattice:
    print('extent', extent, "intent", intent)

c.lattice.graphviz(view=True)
