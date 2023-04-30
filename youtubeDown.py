from pytube import YouTube

DOWNLOAD_FOLDER = "./videos/"
# url = "https://www.youtube.com/watch?v=Z6rJrSeJ7Vg"
url = "http://www.youtube.com/watch?v=2YJ7w1-9BFI"
# yt = YouTube(url)
# yt.streams.download(DOWNLOAD_FOLDER)
YouTube(url).streams.get_highest_resolution().download()