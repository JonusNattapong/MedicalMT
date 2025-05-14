import kagglehub

# Download latest version and save to DatasetDownload/fer2013
path = kagglehub.dataset_download("msambare/fer2013", download_dir="DatasetDownload/fer2013")

print("Path to dataset files:", path)
