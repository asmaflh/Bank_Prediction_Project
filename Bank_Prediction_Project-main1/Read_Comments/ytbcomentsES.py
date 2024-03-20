from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from textblob import TextBlob
import matplotlib.pyplot as plt
import csv




def get_video_comments(video_id, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Get video details
    video_response = youtube.videos().list(
        part='snippet',
        id=video_id
    ).execute()

    video_title = video_response['items'][0]['snippet']['title']

    # Get comments
    comments = []
    nextPageToken = None
    while True:
        try:
            comment_response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=200,  # Adjust as needed
                pageToken=nextPageToken
            ).execute()

            for item in comment_response['items']:
                Text = item['snippet']['topLevelComment']['snippet']['textDisplay']
                commenterID = item['snippet']['topLevelComment']['snippet']['authorChannelId']['value']
                UserName = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
                Date = item['snippet']['topLevelComment']['snippet']['publishedAt']
                comments.append((commenterID , UserName,  Date, Text))

            nextPageToken = comment_response.get('nextPageToken')
            if not nextPageToken:
                break


        except HttpError as e:
            print('An HTTP error %d occurred:\n%s' % (e.resp.status, e.content))
            break

    return comments


def save_comments_to_csv(video_ids, api_key, csv_file_name):
    with open(csv_file_name, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['commenterID', 'UserName',  'Date', 'Text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for video_id in video_ids:
            comments = get_video_comments(video_id, api_key)

            for commenterID, UserName,  Date,  Text in comments:
                writer.writerow({'commenterID': commenterID,
                                 'UserName': UserName,
                                 'Date': Date,
                                 'Text': Text})

    print(f"All comments saved to '{csv_file_name}'.")

if __name__ == "__main__":
    video_ids = ['mYYPELvjbF0', 'C1O6dOCNPeY', 'dtCNKgE8hxE', 'QxtGzA4L1yA', 'ylxJA4eBHOA', 'DGtY0rYG-gs', 'Rmmu46u286Y', 'XlkDZyz8tMc',
                 'n6Shu16LCvI', 'z3JwiZsXQ0A', 'Sy2aJj4Vuv0', 'WMiZ8y-tO5o', '1Mu__oDhIe4', 'K3rlC5GNe3s', 'DfHO8ov8IwI', '9x3TNTEDTs0',
                 'fpbf27Eg-Ow', 'AwlECrjAtNg', 'F5ApJc5sSkk', 'QxtGzA4L1yA', 'RU6fIGX88rA', 'FjKeQiwJ5v4', 'jVuYEFoOTSk', '2H8gNLWaawk',
                 'Sy2aJj4Vuv0', 'JP2HI0dIN6U', '7mbd4QZk4q8', 'y1DlkFThI3Q']
    api_key = 'AIzaSyAd1wa842CTow6QZxM4G_LJeeh6xoHRSFE'
    csv_file_name = 'all_commentsES.csv'
    save_comments_to_csv(video_ids, api_key, csv_file_name)



#
# if __name__ == "__main__":
#     video_id = '2G4NNaRkd2c'  # Replace with your video ID
#     api_key = 'AIzaSyAd1wa842CTow6QZxM4G_LJeeh6xoHRSFE'  # Replace with your API key
#
#
#
#
#     # Get comments and print them
#     video_title, comments = get_video_comments(video_id, api_key)
#
#     print(f"Comments for video '{video_title}':")
#     for idx, (commenter_name, comment_date, comment) in enumerate(comments, start=1):
#         print(f"{idx}. Comment by {commenter_name} on {comment_date} : {comment} ")
#
#
