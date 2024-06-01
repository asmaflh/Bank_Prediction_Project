import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
from statsmodels.tsa.seasonal import seasonal_decompose

dji = pd.read_csv('../DJI.csv')
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
close_result = seasonal_decompose(weekly_Close_dataset['Mean_Close'], model='additive', period=20)

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
############################

# read csv files
eng = pd.read_csv('../../Sentiment_Twitter_Data/TwitterData_Class_EN.csv')
fr = pd.read_csv('../../Sentiment_Twitter_Data/TwitterData_Class_FR.csv')
cn = pd.read_csv('../../Sentiment_Twitter_Data/TwitterData_Class_CN.csv')
de = pd.read_csv('../../Sentiment_Twitter_Data/TwitterData_Class_DE.csv')
es = pd.read_csv('../../Sentiment_Twitter_Data/TwitterData_Class_ES.csv')
it = pd.read_csv('../../Sentiment_Twitter_Data/TwitterData_Class_IT.csv')

###############
eng['Date'] = pd.to_datetime(eng['Date'])
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
result = seasonal_decompose(weekly_polarity_dataset['Mean_Polarity'], model='additive', period=20)

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

weekly_polarity_dataset['Polarity_Trend_en'] = result.trend.values
weekly_polarity_dataset['Week_Number'] = weekly_polarity_dataset['Week_Number'].dt.tz_localize(None)
weekly_Close_dataset['Week_Number'] = weekly_Close_dataset['Week_Number'].dt.tz_localize(None)
merged_df = pd.merge(weekly_polarity_dataset[['Week_Number', 'Polarity_Trend_en']],
                     weekly_Close_dataset[['Week_Number', 'Trend']], on='Week_Number', how='right')
print(merged_df)
##############
fr['Date'] = pd.to_datetime(fr['Date'])
fr = fr.sort_values(by='Date')

# Resample the data on a weekly basis and calculate mean polarity for each week
weekly_polarity = fr.resample('W-Mon', on='Date')['Polarity'].mean()
weekly_polarity.fillna(0, inplace=True)

# Plot the weekly polarity
plt.figure(figsize=(12, 6))
weekly_polarity.plot(marker='o', color='b')
plt.title('Weekly Polarity')
plt.xlabel('Weeks')
plt.ylabel('Average Polarity')
plt.grid(True)
plt.show()

weekly_polarity_dataset = pd.DataFrame({
    'Week_Number': weekly_polarity.index,
    'Mean_Polarity': weekly_polarity.values
})
weekly_polarity_dataset['Week_Number'] = pd.to_datetime(weekly_polarity_dataset['Week_Number'])

# Sort the DataFrame by the 'date' column
weekly_polarity_dataset = weekly_polarity_dataset.sort_values(by='Week_Number')

print(weekly_polarity_dataset)

# Perform seasonal decomposition
result = seasonal_decompose(weekly_polarity_dataset['Mean_Polarity'], model='additive', period=20)

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

weekly_polarity_dataset['Polarity_Trend_fr'] = result.trend.values
weekly_polarity_dataset['Week_Number'] = weekly_polarity_dataset['Week_Number'].dt.tz_localize(None)
merged_df = pd.merge(weekly_polarity_dataset[['Week_Number', 'Polarity_Trend_fr']], merged_df, on='Week_Number',
                     how='right')
print(merged_df)
##############
cn['Date'] = pd.to_datetime(cn['Date'])
cn = cn.sort_values(by='Date')

# Resample the data on a weekly basis and calculate mean polarity for each week
weekly_polarity = cn.resample('W-Mon', on='Date')['Polarity'].mean()
weekly_polarity.fillna(0, inplace=True)

# Plot the weekly polarity
plt.figure(figsize=(12, 6))
weekly_polarity.plot(marker='o', color='b')
plt.title('Weekly Polarity')
plt.xlabel('Weeks')
plt.ylabel('Average Polarity')
plt.grid(True)
plt.show()

weekly_polarity_dataset = pd.DataFrame({
    'Week_Number': weekly_polarity.index,
    'Mean_Polarity': weekly_polarity.values
})
weekly_polarity_dataset['Week_Number'] = pd.to_datetime(weekly_polarity_dataset['Week_Number'])

# Sort the DataFrame by the 'date' column
weekly_polarity_dataset = weekly_polarity_dataset.sort_values(by='Week_Number')

print(weekly_polarity_dataset)

# Perform seasonal decomposition
result = seasonal_decompose(weekly_polarity_dataset['Mean_Polarity'], model='additive', period=20)

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

weekly_polarity_dataset['Polarity_Trend_cn'] = result.trend.values
weekly_polarity_dataset['Week_Number'] = weekly_polarity_dataset['Week_Number'].dt.tz_localize(None)
merged_df = pd.merge(weekly_polarity_dataset[['Week_Number', 'Polarity_Trend_cn']], merged_df, on='Week_Number',
                     how='right')
##############
de['Date'] = pd.to_datetime(de['Date'])
de = de.sort_values(by='Date')

# Resample the data on a weekly basis and calculate mean polarity for each week
weekly_polarity = de.resample('W-Mon', on='Date')['Polarity'].mean()
weekly_polarity.fillna(0, inplace=True)

# Plot the weekly polarity
plt.figure(figsize=(12, 6))
weekly_polarity.plot(marker='o', color='b')
plt.title('Weekly Polarity')
plt.xlabel('Weeks')
plt.ylabel('Average Polarity')
plt.grid(True)
plt.show()

weekly_polarity_dataset = pd.DataFrame({
    'Week_Number': weekly_polarity.index,
    'Mean_Polarity': weekly_polarity.values
})
weekly_polarity_dataset['Week_Number'] = pd.to_datetime(weekly_polarity_dataset['Week_Number'])

# Sort the DataFrame by the 'date' column
weekly_polarity_dataset = weekly_polarity_dataset.sort_values(by='Week_Number')

print(weekly_polarity_dataset)

# Perform seasonal decomposition
result = seasonal_decompose(weekly_polarity_dataset['Mean_Polarity'], model='additive', period=20)

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

weekly_polarity_dataset['Polarity_Trend_de'] = result.trend.values
weekly_polarity_dataset['Week_Number'] = weekly_polarity_dataset['Week_Number'].dt.tz_localize(None)
merged_df = pd.merge(weekly_polarity_dataset[['Week_Number', 'Polarity_Trend_de']], merged_df, on='Week_Number',
                     how='right')
##############
es['Date'] = pd.to_datetime(es['Date'])
es = es.sort_values(by='Date')

# Resample the data on a weekly basis and calculate mean polarity for each week
weekly_polarity = es.resample('W-Mon', on='Date')['Polarity'].mean()
weekly_polarity.fillna(0, inplace=True)

# Plot the weekly polarity
plt.figure(figsize=(12, 6))
weekly_polarity.plot(marker='o', color='b')
plt.title('Weekly Polarity')
plt.xlabel('Weeks')
plt.ylabel('Average Polarity')
plt.grid(True)
plt.show()

weekly_polarity_dataset = pd.DataFrame({
    'Week_Number': weekly_polarity.index,
    'Mean_Polarity': weekly_polarity.values
})
weekly_polarity_dataset['Week_Number'] = pd.to_datetime(weekly_polarity_dataset['Week_Number'])

# Sort the DataFrame by the 'date' column
weekly_polarity_dataset = weekly_polarity_dataset.sort_values(by='Week_Number')

print(weekly_polarity_dataset)

# Perform seasonal decomposition
result = seasonal_decompose(weekly_polarity_dataset['Mean_Polarity'], model='additive', period=20)

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

weekly_polarity_dataset['Polarity_Trend_es'] = result.trend.values
weekly_polarity_dataset['Week_Number'] = weekly_polarity_dataset['Week_Number'].dt.tz_localize(None)
merged_df = pd.merge(weekly_polarity_dataset[['Week_Number', 'Polarity_Trend_es']], merged_df, on='Week_Number',
                     how='right')
##############
it['Date'] = pd.to_datetime(it['Date'])
it = it.sort_values(by='Date')

# Resample the data on a weekly basis and calculate mean polarity for each week
weekly_polarity = it.resample('W-Mon', on='Date')['Polarity'].mean()
weekly_polarity.fillna(0, inplace=True)

# Plot the weekly polarity
plt.figure(figsize=(12, 6))
weekly_polarity.plot(marker='o', color='b')
plt.title('Weekly Polarity')
plt.xlabel('Weeks')
plt.ylabel('Average Polarity')
plt.grid(True)
plt.show()

weekly_polarity_dataset = pd.DataFrame({
    'Week_Number': weekly_polarity.index,
    'Mean_Polarity': weekly_polarity.values
})
weekly_polarity_dataset['Week_Number'] = pd.to_datetime(weekly_polarity_dataset['Week_Number'])

# Sort the DataFrame by the 'date' column
weekly_polarity_dataset = weekly_polarity_dataset.sort_values(by='Week_Number')

print(weekly_polarity_dataset)

# Perform seasonal decomposition
result = seasonal_decompose(weekly_polarity_dataset['Mean_Polarity'], model='additive', period=20)

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

weekly_polarity_dataset['Polarity_Trend_it'] = result.trend.values
weekly_polarity_dataset['Week_Number'] = weekly_polarity_dataset['Week_Number'].dt.tz_localize(None)
merged_df = pd.merge(weekly_polarity_dataset[['Week_Number', 'Polarity_Trend_it']], merged_df, on='Week_Number',
                     how='right')
print(merged_df)
############
# plot in same plotting
fig, ax1 = plt.subplots()

color = 'tab:pink'
ax1.set_xlabel('Weeks')
ax1.set_ylabel('Close_Trend', color=color)
ax1.plot(merged_df['Trend'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:green'
ax2.set_ylabel('Polarity_Trend', color=color)
ax2.plot(merged_df['Polarity_Trend_en'], label='en_polarity', color='black')
ax2.plot(merged_df['Polarity_Trend_es'], label='es_polarity', color='yellow')
ax2.plot(merged_df['Polarity_Trend_de'], label='de_polarity', color='green')
ax2.plot( merged_df['Polarity_Trend_cn'], label='cn_polarity', color='brown')
ax2.plot(merged_df['Polarity_Trend_fr'], label='fr_polarity', color='orange')
ax2.plot(merged_df['Polarity_Trend_it'], label='it_polarity', color='gray')

ax2.tick_params(axis='y', labelcolor=color)
fig.suptitle('Comparison between polarity trend and close trend', fontweight="bold")
ax2.legend(loc='upper left')  # Adjust the location as needed
plt.grid(True)
plt.show()

###############
merged_df['Week_Number'] = merged_df['Week_Number'].dt.strftime('%Y-%m-%d')
merged_df['Week_Number'] = 'Week_' + (merged_df.index + 1).astype(str)
merged_df.set_index('Week_Number', inplace=True)
print(merged_df.index)
print(merged_df.head(10))


def intervals(column):
    interval_size = 12 # Define the size of each interval
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
trend_results = intervals('Polarity_Trend_en')

# Create three new columns based on the result
merged_df['Sen_I_en'] = 0
merged_df['Sen_D_en'] = 0
merged_df['Sen_S_en'] = 0

for start_week, end_week, trend in trend_results:
    if trend == 'Increasing':
        merged_df.loc[start_week:end_week, 'Sen_I_en'] = 1
    elif trend == 'Decreasing':
        merged_df.loc[start_week:end_week, 'Sen_D_en'] = 1
    else:
        merged_df.loc[start_week:end_week, 'Sen_S_en'] = 1
########################
# Apply the intervals function to merged_df
trend_results = intervals('Polarity_Trend_fr')

# Create three new columns based on the result
merged_df['Sen_I_fr'] = 0
merged_df['Sen_D_fr'] = 0
merged_df['Sen_S_fr'] = 0

for start_week, end_week, trend in trend_results:
    if trend == 'Increasing':
        merged_df.loc[start_week:end_week, 'Sen_I_fr'] = 1
    elif trend == 'Decreasing':
        merged_df.loc[start_week:end_week, 'Sen_D_fr'] = 1
    else:
        merged_df.loc[start_week:end_week, 'Sen_S_fr'] = 1
########################
# Apply the intervals function to merged_df
trend_results = intervals('Polarity_Trend_cn')

# Create three new columns based on the result
merged_df['Sen_I_cn'] = 0
merged_df['Sen_D_cn'] = 0
merged_df['Sen_S_cn'] = 0

for start_week, end_week, trend in trend_results:
    if trend == 'Increasing':
        merged_df.loc[start_week:end_week, 'Sen_I_cn'] = 1
    elif trend == 'Decreasing':
        merged_df.loc[start_week:end_week, 'Sen_D_cn'] = 1
    else:
        merged_df.loc[start_week:end_week, 'Sen_S_cn'] = 1
########################
# Apply the intervals function to merged_df
trend_results = intervals('Polarity_Trend_de')

# Create three new columns based on the result
merged_df['Sen_I_de'] = 0
merged_df['Sen_D_de'] = 0
merged_df['Sen_S_de'] = 0

for start_week, end_week, trend in trend_results:
    if trend == 'Increasing':
        merged_df.loc[start_week:end_week, 'Sen_I_de'] = 1
    elif trend == 'Decreasing':
        merged_df.loc[start_week:end_week, 'Sen_D_de'] = 1
    else:
        merged_df.loc[start_week:end_week, 'Sen_S_de'] = 1
########################
# Apply the intervals function to merged_df
trend_results = intervals('Polarity_Trend_es')

# Create three new columns based on the result
merged_df['Sen_I_es'] = 0
merged_df['Sen_D_es'] = 0
merged_df['Sen_S_es'] = 0

for start_week, end_week, trend in trend_results:
    if trend == 'Increasing':
        merged_df.loc[start_week:end_week, 'Sen_I_es'] = 1
    elif trend == 'Decreasing':
        merged_df.loc[start_week:end_week, 'Sen_D_es'] = 1
    else:
        merged_df.loc[start_week:end_week, 'Sen_S_es'] = 1
########################
# Apply the intervals function to merged_df
trend_results = intervals('Polarity_Trend_it')

# Create three new columns based on the result
merged_df['Sen_I_it'] = 0
merged_df['Sen_D_it'] = 0
merged_df['Sen_S_it'] = 0

for start_week, end_week, trend in trend_results:
    if trend == 'Increasing':
        merged_df.loc[start_week:end_week, 'Sen_I_it'] = 1
    elif trend == 'Decreasing':
        merged_df.loc[start_week:end_week, 'Sen_D_it'] = 1
    else:
        merged_df.loc[start_week:end_week, 'Sen_S_it'] = 1
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
merged_df = merged_df.drop(
    ['Polarity_Trend_en', 'Trend', 'Polarity_Trend_es', 'Polarity_Trend_de', 'Polarity_Trend_cn', 'Polarity_Trend_fr',
     'Polarity_Trend_it'], axis=1)
print(merged_df.head(50))
merged_df.to_csv('../Csvs/All_context.csv')
#########################################################
from concepts import Context

bools = merged_df.values.tolist()

c = Context(objects=merged_df.index, properties=merged_df.columns, bools=bools)

print(c.objects)
print(c.properties)
print(c.bools)

for extent, intent in c.lattice:
    print('extent', extent, "intent", intent)

#c.lattice.graphviz(view=True)

