from typing import Union
from pytube import YouTube as yt

links = [
     'https://www.youtube.com/watch?v=4LQLvgXLguk',
     'https://www.youtube.com/watch?v=CIHun1gx7zU',
     'https://www.youtube.com/watch?v=6dSkaRnnc5s',
     'https://www.youtube.com/watch?v=71-tMMtVWMI',
     'https://www.youtube.com/watch?v=sWu7Sok3kok',
     'https://www.youtube.com/watch?v=6UZJfjBcUKM'
]
hidden_hazard = [
     'https://www.youtube.com/watch?v=EwgbjEKO6xE', # police car chase
     'https://www.youtube.com/watch?v=c48IdUzouo0', # lightnings
     'https://www.youtube.com/watch?v=hr8knfSooVI', # explosion
     'https://www.youtube.com/watch?v=cYB_Za5eI3Q', # explosion 2
     'https://www.youtube.com/watch?v=8disaQwktk0' # paparazzi splashes

]
def download_video(url:str,path:str = 'videos/', filename:Union[str,None] = None)->bool:
     '''

     '''
     video = yt(url)
     stream = video.streams.get_highest_resolution()
     try:
          stream.download(path)
     except Exception:
          return False
     return True
if __name__ == "__main__":
     for link in links:
          download_video(link)

     for link in hidden_hazard:
          download_video(link)