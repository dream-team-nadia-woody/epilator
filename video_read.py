from pytube import YouTube as yt

video = yt('https://www.youtube.com/watch?v=4LQLvgXLguk')
video = video.streams.get_highest_resolution()
video.download('videos/')