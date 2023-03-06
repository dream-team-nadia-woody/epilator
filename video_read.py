from typing import Union
from pytube import YouTube as yt

def download_video(url:str,path:str = 'videos/', filename:Union[str,None] = None)->bool:
    video = yt(url)
    stream = video.streams.get_highest_resolution()
    try:
         stream.download(path)
    except Exception:
         return False
    return True

