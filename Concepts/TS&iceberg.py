import xml.etree.ElementTree as ET
import pandas as pd
import pytz
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
# Parse the XML file
tree = ET.parse('Yb&Tw_FRcontext.lat.xml')
root = tree.getroot()
lattice_elem = root.find("Lattice")
nb_concept = lattice_elem.get("numberCpt")
conceptId_ref = []
lattice = []

# Find the concept with ID equal to nb_concept
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
                "Id": cp,
                "Extent": week,
                "Intent": prop
            }
            lattice.append(item)
            break
    else:
        print(f"Concept with ID {cp} not found")

for lat in lattice:
    lat["Extent"].sort()

df = pd.DataFrame(lattice)
print(df)

# Getting the dates
dji = pd.read_csv('../Time_series_dec/DJI.csv')
dji['Date'] = pd.to_datetime(dji['Date'])
dji = dji.sort_values(by='Date')

weekly_Close = dji.resample('W-Mon', on='Date')['Close'].mean()
weekly_Close.fillna(0, inplace=True)

weekly_dataset = pd.DataFrame({
    'Week_Number': weekly_Close.index,
})
weekly_dataset['Week_Number'] = pd.to_datetime(weekly_dataset['Week_Number'])
weekly_dataset = weekly_dataset.sort_values(by='Week_Number')
weekly_dataset['Week_index'] = 'Week_' + (weekly_dataset.index).astype(str)
weekly_dataset = weekly_dataset.reset_index()


def extract_dates(concept_id):
    dates = pd.DataFrame(columns=['index', 'Date'])
    list_index = []
    list_dates = []
    for lat in lattice:
        if concept_id == lat["Id"]:
            for week in lat["Extent"]:
                for index, row in weekly_dataset.iterrows():
                    if week == row['Week_index']:
                        list_index.append(index)
                        list_dates.append(row['Week_Number'])
            list_index.sort()
            list_dates.sort()
            dates = pd.DataFrame({
                'index': list_index,
                'Date': list_dates
            })
            dates.set_index('index', inplace=True)
            break
    return dates


def extract_info(df, first, last, idtext, dates):
    text_id = []
    dji_close = []

    if first in dates.index and last in dates.index:
        for index, row in dji.iterrows():
            if dates.loc[first, 'Date'].replace(tzinfo=pytz.UTC) <= row['Date'].replace(tzinfo=pytz.UTC) <= dates.loc[
                last, 'Date'].replace(tzinfo=pytz.UTC):
                dji_close.append(row["Close"])

        for index, row in df.iterrows():
            if dates.loc[first, 'Date'].replace(tzinfo=pytz.UTC) <= row['Date'].replace(tzinfo=pytz.UTC) <= dates.loc[
                last, 'Date'].replace(tzinfo=pytz.UTC):
                text_id.append(row[idtext])
    else:
        print(f"Indices {first} and/or {last} not found in dates DataFrame.")

    return dji_close, text_id


# Extract the concept by its id
dates =extract_dates("34")
for i, index in enumerate(dates.index):
    print(index)
print(dates.values)

first = 265
last = 312
df = pd.read_csv('../Twitter_Datasets/TwitterData_EN.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.tz_convert('UTC')
df = df.sort_values(by='Date').reset_index()
close, text_id = extract_info(df, first, last, "TweetId", dates)
print("Dates between ", dates.loc[first, 'Date'].replace(tzinfo=pytz.UTC), "and ", dates.loc[
    last, 'Date'].replace(tzinfo=pytz.UTC))
print("the close price in this period")
for c in close:
    print(c)
print("the comments ids in this period")
for id in text_id:
    print(id)

df2 = pd.read_csv('../Sentiment_Twitter_Data/TwitterData_Class_EN.csv')
df2['Date'] = pd.to_datetime(df2['Date'])
df2 = df2.sort_values(by='Date')
average_polarity = df2.groupby('Date')['Polarity'].mean()

# Convert timezone-aware datetimes to timezone-naive
dates['Date'] = dates['Date'].dt.tz_localize(None)
average_polarity.index = average_polarity.index.tz_localize(None)
dji['Date'] = dji['Date'].dt.tz_localize(None)

# Plotting the two datasets
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('dates')
ax1.set_ylabel('Polarity', color=color)
ax1.plot(average_polarity.index, average_polarity.values, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Create a mask for the specified date range
mask = (average_polarity.index >= dates.loc[first, 'Date']) & \
       (average_polarity.index <= dates.loc[last, 'Date'])

# Fill the area within the specified date range
ax1.fill_between(average_polarity.index, average_polarity.values, where=mask, color='black', alpha=0.4)

ax2 = ax1.twinx()

color = 'tab:green'
ax2.set_ylabel('Close', color=color)
ax2.plot(dji['Date'], dji['Close'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Create a mask for the specified date range
mask = (dji['Date'] >= dates.loc[first, 'Date']) & \
       (dji['Date'] <= dates.loc[last, 'Date'])

# Fill the area within the specified date range
ax2.fill_between(dji['Date'], dji['Close'], where=mask, color='black', alpha=0.4)

fig.suptitle('Comparison between Polarity Trend and Close Trend', fontweight="bold")
plt.show()