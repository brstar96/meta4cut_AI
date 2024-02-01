from pytube import YouTube

# Read the file containing the YouTube video URLs
with open("/projects/meta4cut_BE/preprocessors/videodownloader/urls.txt") as f:
    urls = f.readlines()

# Download each video
for url in urls:
    # Remove any leading or trailing whitespaces from the URL
    url = url.strip()
    
    # Create a YouTube object and get the stream
    yt = YouTube(url)
    stream = yt.streams.filter(file_extension='mp4', adaptive=True).first()
    
    # Get the filename
    filename = yt.title + "." + stream.mime_type.split("/")[-1]
    
    # Download the video
    print("Downloading: " + filename)
    stream.download(output_path="videos", filename=filename)
    print("Video downloaded successfully!")