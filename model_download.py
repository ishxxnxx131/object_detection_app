import urllib.request

urls = [
    "https://pjreddie.com/media/files/yolov3.weights",
    "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
]

for url in urls:
    file_name = url.split("/")[-1]
    print(f"Downloading {file_name}...")
    urllib.request.urlretrieve(url, file_name)
    print(f"{file_name} downloaded successfully!")
