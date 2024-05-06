import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.collections import LineCollection
import numpy as np

# Parse the XML file
tree = ET.parse('All_context.lat.xml')
root = tree.getroot()
lattice = root.find("Lattice")
nb_concept = lattice.get("numberCpt")
conceptId_ref = []
lattice = []
# Find the concept with ID equal to 11
for concept in root.iter('Concept'):
    concept_id = concept.find('ID')
    if concept_id is not None and concept_id.text.strip() == nb_concept:
        # Get the UpperCovers element of the concept
        uppercovers = concept.find('UpperCovers')
        if uppercovers is not None:
            concept_refs = uppercovers.findall('Concept_Ref')
            for concept_ref in concept_refs:
                print(concept_ref.text)
                conceptId_ref.append(concept_ref.text)
        else:
            print(f"UpperCovers element not found for concept with ID {nb_concept}")
        break
else:
    print(f"Concept with ID {nb_concept} not found")
for cp in conceptId_ref:
    for concept in root.iter('Concept'):
        concept_id = concept.find('ID')
        if concept_id is not None and concept_id.text.strip() == cp:
            week = []
            prop = []
            # Get the UpperCovers element of the concept
            extent = concept.find('Extent')
            intent = concept.find('Intent')
            if extent is not None:
                object_refs = extent.findall('Object_Ref')
                for object_ref in object_refs:
                    week.append(object_ref.text)
            else:
                print(f"object element not found for concept with ID {nb_concept}")
            if intent is not None:
                attr_refs = intent.findall('Attribute_Ref')
                for attr_ref in attr_refs:
                    prop.append(attr_ref.text)
            else:
                print(f"attribute element not found for concept with ID {nb_concept}")
            item = {
                "Weeks": week,
                "Prop": prop
            }
            lattice.append(item)
            break
    else:
        print(f"Concept with ID {nb_concept} not found")
for lat in lattice:
    print(lat["Weeks"])
    print(lat["Prop"])

# ###################
#
# eng = pd.read_csv('Sentiment_Twitter_Data/TwitterData_Class_EN.csv')
# # Convert 'date' column to datetime
# eng['Date'] = pd.to_datetime(eng['Date'])
# # Sort the DataFrame by the 'date' column
# eng = eng.sort_values(by='Date')
#
# # Resample the data on a weekly basis and calculate mean polarity for each week
# weekly_polarity = eng.resample('W-Mon', on='Date')['Polarity'].mean()
# weekly_polarity.fillna(0, inplace=True)
# # Create a new DataFrame containing week number and mean polarity
# weekly_polarity_dataset = pd.DataFrame({
#     'Week_Number': weekly_polarity.index,  # week number
#     'Mean_Polarity': weekly_polarity.values  # Use the mean polarity values
# })
# # Convert 'date' column to datetime
# weekly_polarity_dataset['Week_Number'] = pd.to_datetime(weekly_polarity_dataset['Week_Number'])
#
# # Sort the DataFrame by the 'date' column
# weekly_polarity_dataset = weekly_polarity_dataset.sort_values(by='Week_Number')
#
# print(weekly_polarity_dataset)
#
# # Perform seasonal decomposition
# result = seasonal_decompose(weekly_polarity_dataset['Mean_Polarity'], model='additive', period=20)
#
# weekly_polarity_dataset['Polarity_Trend'] = result.trend.values
#
# ############################
#
# # read csv files
# dji = pd.read_csv('Time_series_dec/DJI.csv')
# dji['Date'] = pd.to_datetime(dji['Date'])
# dji = dji.sort_values(by='Date')
#
# weekly_Close = dji.resample('W-Mon', on='Date')['Close'].mean()
# weekly_Close.fillna(0, inplace=True)
#
# weekly_Close_dataset = pd.DataFrame({
#     'Week_Number': weekly_Close.index,
#     'Mean_Close': weekly_Close.values
# })
# weekly_Close_dataset['Week_Number'] = pd.to_datetime(weekly_Close_dataset['Week_Number'])
#
# weekly_Close_dataset = weekly_Close_dataset.sort_values(by='Week_Number')
#
# print(weekly_Close_dataset.head(420))
#
# # Perform seasonal decomposition
# close_result = seasonal_decompose(weekly_Close_dataset['Mean_Close'], model='additive', period=20)
#
# # add the trend as column
# weekly_Close_dataset['Trend'] = close_result.trend.values
#
# print(weekly_Close_dataset.head(10))
#
# # merge the two datasets
# weekly_polarity_dataset['Week_Number'] = weekly_polarity_dataset['Week_Number'].dt.tz_localize(None)
# weekly_Close_dataset['Week_Number'] = weekly_Close_dataset['Week_Number'].dt.tz_localize(None)
# merged_df = pd.merge(weekly_polarity_dataset, weekly_Close_dataset, on='Week_Number', how='right')
# ###############
# merged_df['Week_index'] = merged_df['Week_Number'].dt.strftime('%Y-%m-%d')
# merged_df['Week_index'] = 'Week_' + (merged_df.index + 1).astype(str)
# merged_df = merged_df.reset_index()
# list_index = []
# list_dates = []
# for week in lattice[2]["Weeks"]:
#     for index, row in merged_df.iterrows():
#         if week == row['Week_index']:
#             print(index)
#             list_index.append(index)
#             list_dates.append(row['Week_Number'])
# list_index.sort()
# list_dates.sort()
# print(list_index)
# print(list_dates)
# print(merged_df)
#
# # #############
# # plot in same plotting
# fig, ax1 = plt.subplots()
# Y = merged_df['Week_Number']
# color = 'tab:blue'
# ax1.set_xlabel('Weeks')
# ax1.set_ylabel('Polarity Trend', color=color)
# ax1.plot(Y[0:140], merged_df['Polarity_Trend'][0:140], color=color)
# ax1.plot(Y[140:159], merged_df['Polarity_Trend'][140:159], color="yellow")
# ax1.plot(Y[159:], merged_df['Polarity_Trend'][159:], color=color)
# ax1.tick_params(axis='y', labelcolor=color)
#
# ax2 = ax1.twinx()
#
# color = 'tab:green'
# ax2.set_ylabel('Close trend', color=color)
# ax2.plot(Y[0:260], merged_df['Trend'][0:260], color=color)
# ax2.plot(Y[260:300], merged_df['Trend'][260:300], color="yellow")
# ax2.plot(Y[300:], merged_df['Trend'][300:], color=color)
# ax2.tick_params(axis='y', labelcolor=color)
# fig.suptitle('comparision between polarity trend and close trend', fontweight="bold")
# plt.show()
#
# tweets = pd.read_csv('Twitter_Datasets/TwitterData_EN.csv')
# tweets['Date'] = pd.to_datetime(tweets['Date'])
# tweets['Date'] = tweets['Date'].dt.tz_convert('UTC')
#
# tweets = tweets.sort_values(by='Date')
# print(tweets.Date)
# tweets = tweets.reset_index()
# import pytz
#
# # Convert the timezone-naive timestamps to timezone-aware timestamps in UTC
# list_dates = [date.replace(tzinfo=pytz.UTC) for date in list_dates]
#
# for index, row in tweets.iterrows():
#     if list_dates[120] <= row['Date'].replace(tzinfo=pytz.UTC) <= list_dates[150]:
#         print(row['Date'])
#         print(row['Tweet'])
#         print(row['TweetId'])
#
#
