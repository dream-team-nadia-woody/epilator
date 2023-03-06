from typing import Union
from pytube import YouTube as yt

links = [
     'https://www.youtube.com/watch?v=4LQLvgXLguk',
     'https://www.youtube.com/watch?v=CIHun1gx7zU',
     'https://www.youtube.com/watch?v=6dSkaRnnc5s',
     'https://www.youtube.com/watch?v=71-tMMtVWMI&t=17s',
     'https://www.youtube.com/watch?v=sWu7Sok3kok',
     'https://www.youtube.com/watch?v=6UZJfjBcUKM'
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

for link in links:
     download_video(link)