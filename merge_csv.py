import pandas as pd

# # Read CSV files
# df1 = pd.read_csv('TwitterData2012.csv')
# df2 = pd.read_csv('TwitterData.csv')
#
# # Merge DataFrames
#
# merged_df =df1.merge(df2,how="outer")
# # Save Merged DataFrame to CSV
# merged_df.to_csv('TwitterData_EN.csv', index=False)
df1 = pd.read_csv('Twitter_Datasets/TwitterData_EN.csv')
df1=df1.drop('Unnamed: 0',axis=1)
df1.to_csv('TwitterData_EN.csv', index=False)